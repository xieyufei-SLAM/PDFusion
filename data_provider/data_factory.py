from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
    Dataset_Custom_NoCorrection, Dataset_Custom_NoCorrection_Calce, Dataset_Custom_Calce
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'custom_NoCorrection': Dataset_Custom_NoCorrection,
    'custom_NoCorrection_Calce': Dataset_Custom_NoCorrection_Calce,
    'custom_Calce': Dataset_Custom_Calce,
}

def data_provider(args, flag, Domain_index):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if Domain_index == 'source':
        args.data_path = args.source_path
    elif Domain_index == 'target':
        args.data_path = args.target_path
    else:
        args.data_path = args.data_path

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
