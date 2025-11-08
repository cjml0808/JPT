from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
from .mask import Mask


class CCTrainer(object):
    def __init__(self, args, encoder, memory=None):
        super(CCTrainer, self).__init__()
        self.args = args
        self.encoder = encoder
        self.memory = memory
        self.criterion = torch.nn.CrossEntropyLoss().to(torch.device('cuda'))

    def gatherFeatures(self, features, local_rank, world_size):
        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[local_rank] = features
        features = torch.cat(features_list)
        return features

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(int(features.shape[0]/2)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(torch.device('cuda'))

        features = torch.nn.functional.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(torch.device('cuda'))
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # 考虑到前面的label是（0,1,2,...,255,0,1,2,...,255）的形式，label.bool()中应该还是有非0数（也就是1）的，只不过矩阵太大了，看不出来

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # 在构造logits张量时，使第一个元素（索引为0的列）对应于正样本的相似度，而其余的元素对应于负样本的相似度
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(torch.device('cuda'))
        # 在logits中正样本对在第1列，其标签或索引视为0（或等价的，将负样本的索引视为大于0的整数）

        logits = logits / 0.07
        return logits, labels

    def train(self, epoch, data_loader, unclustered_loader, optimizer, print_freq=10, train_iters=400, alpha=0.5):
        pruneMask = Mask(self.encoder)
        prunePercent = self.args.prune_percent
        randomPrunePercent = self.args.random_prune_percent
        magnitudePrunePercent = prunePercent - randomPrunePercent

        print("current prune percent is {}".format(prunePercent))
        if randomPrunePercent > 0:
            print("random prune percent is {}".format(randomPrunePercent))

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()

        # prune every epoch
        pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            unclustered_inputs = unclustered_loader.next()
            # unclustered_inputs = unclustered_loader[0]
            data_time.update(time.time() - end)
            # process inputs
            inputs, labels = self._parse_data(inputs)
            unclustered_inputs, _ = self._parse_data(unclustered_inputs)
            # unclustered_inputs = torch.cat(unclustered_inputs, dim=0)
            # unclustered_out = self._forward(unclustered_inputs)
            with torch.no_grad():
                self.encoder.module.resnet.set_prune_flag(True)
                features_2_noGrad = self._forward(unclustered_inputs[1])
                # features_2_noGrad = self.gatherFeatures(features_2, 1, 1).detach()
            self.encoder.module.resnet.set_prune_flag(False)
            features_1 = self._forward(unclustered_inputs[0])
            # features_1 = self.gatherFeatures(features_1, 1, 1)
            unclustered_out = torch.cat([features_1, features_2_noGrad], dim=0)
            unclustered_logits, unclustered_labels = self.info_nce_loss(unclustered_out)
            unclustered_loss = self.criterion(unclustered_logits, unclustered_labels)

            # forward
            f_out = self._forward(inputs)
            clustered_loss = self.memory(f_out, labels)

            loss = alpha * clustered_loss + (1-alpha) * unclustered_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            loss_val = loss.detach().item()

            # calculate the grad for pruned network
            features_1_no_grad = features_1.detach()
            self.encoder.module.resnet.set_prune_flag(True)
            features_2 = self._forward(unclustered_inputs[1])
            # features_2 = self.gatherFeatures(features_2, 1, 1)
            unclustered_out = torch.cat([features_1_no_grad, features_2], dim=0)
            unclustered_logits, unclustered_labels = self.info_nce_loss(unclustered_out)
            unclustered_loss = self.criterion(unclustered_logits, unclustered_labels)
            loss = alpha * clustered_loss + (1 - alpha) * unclustered_loss
            loss.backward()

            optimizer.step()

            losses.update(loss_val)

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, indexes = inputs
        if isinstance(imgs, list):
            return [imgs[0].cuda(), imgs[1].cuda()], pids.cuda()
        else:
            return imgs.cuda(), pids.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

