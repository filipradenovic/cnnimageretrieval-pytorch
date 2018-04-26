import os

def download_test(data_dir):
    """
    DOWNLOAD_TEST Checks, and, if required, downloads the necessary datasets for the testing.
      
        download_test(DATA_ROOT) checks if the data necessary for running the example script exist.
        If not it downloads it in the folder structure:
            DATA_ROOT/test/oxford5k/  : folder with Oxford images and ground truth file
            DATA_ROOT/test/paris6k/   : folder with Paris images and ground truth file
            DATA_ROOT/test/roxford5k/ : folder with Oxford images and revisited ground truth file
            DATA_ROOT/test/rparis6k/  : folder with Paris images and revisited ground truth file
    """

    # Create data folder if it does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # Create datasets folder if it does not exist
    datasets_dir = os.path.join(data_dir, 'test')
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)

    # Download datasets folders test/DATASETNAME/
    datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
    for di in range(len(datasets)):
        dataset = datasets[di]

        if dataset == 'oxford5k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images.tgz']
        elif dataset == 'paris6k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1.tgz', 'paris_2.tgz']
        elif dataset == 'roxford5k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
            dl_files = ['oxbuild_images.tgz']
        elif dataset == 'rparis6k':
            src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/parisbuildings'
            dl_files = ['paris_1.tgz', 'paris_2.tgz']
        else:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        dst_dir = os.path.join(datasets_dir, dataset, 'jpg')
        if not os.path.isdir(dst_dir):

            # for oxford and paris download images
            if dataset == 'oxford5k' or dataset == 'paris6k':
                print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset, dst_dir))
                os.makedirs(dst_dir)
                for dli in range(len(dl_files)):
                    dl_file = dl_files[dli]
                    src_file = os.path.join(src_dir, dl_file)
                    dst_file = os.path.join(dst_dir, dl_file)
                    print('>> Downloading dataset {} archive {}...'.format(dataset, dl_file))
                    os.system('wget {} -O {}'.format(src_file, dst_file))
                    print('>> Extracting dataset {} archive {}...'.format(dataset, dl_file))
                    # create tmp folder
                    dst_dir_tmp = os.path.join(dst_dir, 'tmp')
                    os.system('mkdir {}'.format(dst_dir_tmp))
                    # extract in tmp folder
                    os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
                    # remove all (possible) subfolders by moving only files in dst_dir
                    os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
                    # remove tmp folder
                    os.system('rm -rf {}'.format(dst_dir_tmp))
                    print('>> Extracted, deleting dataset {} archive {}...'.format(dataset, dl_file))
                    os.system('rm {}'.format(dst_file))

            # for roxford and rparis just make sym links
            elif dataset == 'roxford5k' or dataset == 'rparis6k':
                print('>> Dataset {} directory does not exist. Creating: {}'.format(dataset, dst_dir))
                dataset_old = dataset[1:]
                dst_dir_old = os.path.join(datasets_dir, dataset_old, 'jpg')
                os.mkdir(os.path.join(datasets_dir, dataset))
                os.system('ln -s {} {}'.format(dst_dir_old, dst_dir))
                print('>> Created symbolic link from {} jpg to {} jpg'.format(dataset_old, dataset))


        gnd_src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'test', dataset)
        gnd_dst_dir = os.path.join(datasets_dir, dataset)
        gnd_dl_file = 'gnd_{}.pkl'.format(dataset)
        gnd_src_file = os.path.join(gnd_src_dir, gnd_dl_file)
        gnd_dst_file = os.path.join(gnd_dst_dir, gnd_dl_file)
        if not os.path.exists(gnd_dst_file):
            print('>> Downloading dataset {} ground truth file...'.format(dataset))
            os.system('wget {} -O {}'.format(gnd_src_file, gnd_dst_file))


def download_train(data_dir):
    """
    DOWNLOAD_TRAIN Checks, and, if required, downloads the necessary datasets for the training.
      
        download_train(DATA_ROOT) checks if the data necessary for running the example script exist.
        If not it downloads it in the folder structure:
            DATA_ROOT/train/retrieval-SfM-120k/  : folder with rsfm120k images and db files
            DATA_ROOT/train/retrieval-SfM-30k/   : folder with rsfm30k images and db files
    """

    # Create data folder if it does not exist
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    # Create datasets folder if it does not exist
    datasets_dir = os.path.join(data_dir, 'train')
    if not os.path.isdir(datasets_dir):
        os.mkdir(datasets_dir)

    # Download folder train/retrieval-SfM-120k/
    src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'ims')
    dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
    dl_file = 'ims.tar.gz'
    if not os.path.isdir(dst_dir):
        src_file = os.path.join(src_dir, dl_file)
        dst_file = os.path.join(dst_dir, dl_file)
        print('>> Image directory does not exist. Creating: {}'.format(dst_dir))
        os.makedirs(dst_dir)
        print('>> Downloading ims.tar.gz...')
        os.system('wget {} -O {}'.format(src_file, dst_file))
        print('>> Extracting {}...'.format(dst_file))
        os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir))
        print('>> Extracted, deleting {}...'.format(dst_file))
        os.system('rm {}'.format(dst_file))

    # Create symlink for train/retrieval-SfM-30k/ 
    dst_dir_old = os.path.join(datasets_dir, 'retrieval-SfM-120k', 'ims')
    dst_dir = os.path.join(datasets_dir, 'retrieval-SfM-30k', 'ims')
    if not os.path.isdir(dst_dir):
        os.makedirs(os.path.join(datasets_dir, 'retrieval-SfM-30k'))
        os.system('ln -s {} {}'.format(dst_dir_old, dst_dir))
        print('>> Created symbolic link from retrieval-SfM-120k/ims to retrieval-SfM-30k/ims')

    # Download db files
    src_dir = os.path.join('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'dbs')
    datasets = ['retrieval-SfM-120k', 'retrieval-SfM-30k']
    for dataset in datasets:
        dst_dir = os.path.join(datasets_dir, dataset)
        if dataset == 'retrieval-SfM-120k':
            dl_files = ['{}.pkl'.format(dataset), '{}-whiten.pkl'.format(dataset)]
        elif dataset == 'retrieval-SfM-30k':
            dl_files = ['{}-whiten.pkl'.format(dataset)]

        if not os.path.isdir(dst_dir):
            print('>> Dataset directory does not exist. Creating: {}'.format(dst_dir))
            os.mkdir(dst_dir)

        for i in range(len(dl_files)):
            src_file = os.path.join(src_dir, dl_files[i])
            dst_file = os.path.join(dst_dir, dl_files[i])
            if not os.path.isfile(dst_file):
                print('>> DB file {} does not exist. Downloading...'.format(dl_files[i]))
                os.system('wget {} -O {}'.format(src_file, dst_file))
