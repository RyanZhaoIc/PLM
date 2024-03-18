import os
from copy import deepcopy

import numpy as np
import torch.nn.functional as F
import torch.utils.data

from models.models import get_model
from utils.utils_algo import accuracy_check, one_hot, init_gpuseed, \
    accuracy_check_noise
from utils.utils_algo import get_scheduler
from utils.utils_data import get_origin_datasets, five_cut_transform, indices_split, \
    get_transform, get_noise_dataset, BalancedSampler


def prenp(args, paths, noise_labels):
    if os.path.exists(paths['multi_labels']) and os.path.exists(paths['pre_model']):
        return

    # seed and device
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    # transform
    train_transform = get_transform(dataname=args.ds, train=True)
    val_transform = get_transform(dataname=args.ds, train=False)
    crop_transform = five_cut_transform(args.ds, crop_ratio=args.crop_ratio)

    # dataset
    ordinary_train_dataset, test_dataset, num_classes = get_origin_datasets(dataname=args.ds, transform=val_transform,
                                                                            data_root=args.data_root)

    train_indices, val_indices = indices_split(len_dataset=len(ordinary_train_dataset),
                                               seed=args.seed,
                                               val_ratio=0.1)

    # noise dataset & dataloader
    train_dataset = get_noise_dataset(dataset=ordinary_train_dataset,
                                      noise_labels=noise_labels,
                                      transformations=train_transform,
                                      indices=train_indices)
    es_crop_dataset = get_noise_dataset(dataset=ordinary_train_dataset,
                                        noise_labels=noise_labels,
                                        transformations=crop_transform)
    es_dataset = get_noise_dataset(dataset=ordinary_train_dataset,
                                   noise_labels=noise_labels,
                                   transformations=val_transform,
                                   indices=train_indices)
    val_dataset = get_noise_dataset(dataset=ordinary_train_dataset,
                                    noise_labels=noise_labels,
                                    transformations=val_transform,
                                    indices=val_indices)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True,
                                               persistent_workers=True)
    estimate_dataloader = torch.utils.data.DataLoader(dataset=es_dataset, batch_size=args.bs,
                                                      shuffle=False,
                                                      num_workers=args.num_workers,
                                                      pin_memory=True,
                                                      persistent_workers=True)
    estimate_crop_dataloader = torch.utils.data.DataLoader(dataset=es_crop_dataset, batch_size=args.bs,
                                                           shuffle=False,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.bs,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # labeling
    labels_multi = np.zeros([len(ordinary_train_dataset), num_classes])
    outputs_all = torch.zeros(len(es_dataset), num_classes).to(device)

    # model
    model = get_model(model_name=args.mo, num_classes=num_classes, fix_backbone=False)
    model = model.to(device)
    best_model = deepcopy(model)

    # opt
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum,
                                nesterov=args.nesterov)
    scheduler = get_scheduler(args.ds, optimizer, args.ep)

    # train
    val_accuracy_best = 0
    if args.ds == 'clothing1m':
        b_sampler = BalancedSampler(num_classes, np.array(noise_labels[train_indices]))
    for epoch in range(args.ep):
        if args.ds == 'clothing1m':
            sampled_indices = b_sampler.ind_unsampling()
            term_indices = np.array(train_indices)[sampled_indices].tolist()
            train_dataset = get_noise_dataset(dataset=ordinary_train_dataset,
                                              noise_labels=noise_labels,
                                              transformations=train_transform,
                                              indices=term_indices)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True,
                                                       num_workers=args.num_workers, pin_memory=True,
                                                       persistent_workers=True)
        model.train()
        for images, labels, inds in train_loader:
            X = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.int64, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(X)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

        scheduler.step()
        model.eval()

        val_accuracy = accuracy_check_noise(loader=val_dataloader, model=model, device=device)
        test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
        print('Epoch: {}. Val_Acc: {}. Test_Acc: {}.'.format(epoch, val_accuracy, test_accuracy))

        if val_accuracy_best < val_accuracy:
            val_accuracy_best = val_accuracy
            best_model = deepcopy(model)

    best_model.eval()
    with torch.no_grad():
        for images, labels, inds in estimate_crop_dataloader:
            bs, ncrops, c, h, w = images.size()
            images = images.to(device, non_blocking=True)
            probs = best_model(images.view(-1, c, h, w))
            probs_ncrops = probs.view(bs, ncrops, -1)

            labels_temp = (probs_ncrops == probs_ncrops.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
            labels_temp = labels_temp.sum(1)
            labels_multi[inds, :] = labels_temp.cpu().numpy()

    with torch.no_grad():
        for images, labels, inds in estimate_dataloader:
            images = images.to(device, non_blocking=True)
            outputs = best_model(images)
            outputs_all[inds, :] = outputs
    probs_all = F.softmax(outputs_all, dim=1)

    labels_multi[labels_multi > 1] = 1
    np.save(paths['pre_probs'], probs_all.cpu().numpy())
    np.save(paths['multi_labels'], labels_multi)
    torch.save(best_model.state_dict(), paths['pre_model'])

    noise_oh = one_hot(noise_labels)
    labels_multi = labels_multi + noise_oh
    labels_multi[labels_multi > 1] = 1
    print("Hit Ratio:",
          (np.array([labels_multi[i][ordinary_train_dataset.targets[i]] for i in range(len(labels_multi))]).mean()))
    print("Mean Labels:", labels_multi.mean() * labels_multi.shape[1])
