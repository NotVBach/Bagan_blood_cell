"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import os
import numpy as np
from collections import defaultdict
from optparse import OptionParser
import balancing_gan as bagan
from batch_generator import BatchGenerator
from utils import save_image_array

if __name__ == '__main__':
    argParser = OptionParser()
    argParser.add_option("-u", "--unbalance", default=0.2, type="float", dest="unbalance")
    argParser.add_option("-s", "--random_seed", default=0, type="int", dest="seed")
    argParser.add_option("-d", "--dratio_mode", default="rebalance", type="string", dest="dratio_mode")
    argParser.add_option("-g", "--gratio_mode", default="rebalance", type="string", dest="gratio_mode")
    argParser.add_option("-e", "--epochs", default=50, type="int", dest="epochs")
    argParser.add_option("-l", "--learning_rate", default=0.00005, type="float", dest="adam_lr")
    argParser.add_option("-c", "--target_class", default=-1, type="int", dest="target_class")
    argParser.add_option("-D", "--dataset_dir", default="noaug", type="string", dest="dataset_dir")

    (options, args) = argParser.parse_args()
    assert 0.0 < options.unbalance <= 1.0, "Unbalance factor must be > 0 and <= 1"

    np.random.seed(options.seed)
    dataset_dir = options.dataset_dir
    batch_size = 32
    image_shape = [3, 32, 32]

    res_dir = f"res_defects_dmode_{options.dratio_mode}_gmode_{options.gratio_mode}_unbalance_{options.unbalance}_epochs_{options.epochs}_lr_{options.adam_lr}_seed_{options.seed}"
    os.makedirs(res_dir, exist_ok=True)

    bg_train_full = BatchGenerator(BatchGenerator.TRAIN, batch_size, dataset_dir=dataset_dir)
    bg_test = BatchGenerator(BatchGenerator.TEST, batch_size, dataset_dir=dataset_dir)
    bg_val = BatchGenerator(BatchGenerator.VAL, batch_size, dataset_dir=dataset_dir)

    min_latent_res = image_shape[-1]
    while min_latent_res > 8:
        min_latent_res /= 2
    min_latent_res = int(min_latent_res)

    classes = bg_train_full.get_label_table()
    img_samples = defaultdict(list)

    target_classes = classes
    if options.target_class >= 1:
        min_classes = [options.target_class]
    else:
        min_classes = target_classes

    for c in min_classes:
        if options.unbalance == 1.0 and c > min_classes[0] and all(
            os.path.exists(f"{res_dir}/class_{min_classes[0]}_{suffix}") for suffix in
            ["score.csv", "discriminator.weights.h5", "generator.weights.h5", "reconstructor.weights.h5"]
        ):
            for suffix in ["score.csv", "discriminator.weights.h5", "generator.weights.h5", "reconstructor.weights.h5"]:
                os.symlink(f"{res_dir}/class_{min_classes[0]}_{suffix}", f"{res_dir}/class_{c}_{suffix}")

        bg_train_partial = BatchGenerator(BatchGenerator.TRAIN, batch_size, class_to_prune=c, unbalance=options.unbalance, dataset_dir=dataset_dir)

        if not all(os.path.exists(f"{res_dir}/class_{c}_{suffix}") for suffix in ["score.csv", "discriminator.weights.h5", "generator.weights.h5", "reconstructor.weights.h5"]):
            print(f"Training BAGAN for class {c}")
            gan = bagan.BalancingGAN(
                target_classes, c, dratio_mode=options.dratio_mode, gratio_mode=options.gratio_mode,
                adam_lr=options.adam_lr, res_dir=res_dir, image_shape=image_shape, min_latent_res=min_latent_res
            )
            gan.train(bg_train_partial, bg_val, epochs=options.epochs)
            gan.save_history(res_dir, c)
        else:
            print(f"Loading BAGAN for class {c}")
            gan = bagan.BalancingGAN(
                target_classes, c, dratio_mode=options.dratio_mode, gratio_mode=options.gratio_mode,
                adam_lr=options.adam_lr, res_dir=res_dir, image_shape=image_shape, min_latent_res=min_latent_res
            )
            gan.load_models(
                f"{res_dir}/class_{c}_generator.weights.h5",
                f"{res_dir}/class_{c}_discriminator.weights.h5",
                f"{res_dir}/class_{c}_reconstructor.weights.h5",
                bg_train=bg_train_partial
            )

        img_samples[f'class_{c}'] = gan.generate_samples(c=c, samples=10)
        save_image_array(np.array([img_samples[f'class_{c}']]), f'{res_dir}/plot_class_{c}.png')