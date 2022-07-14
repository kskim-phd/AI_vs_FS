import pickle


def save_pickle(pickle_path, arg):
    f = open(pickle_path, 'wb')
    pickle.dump(arg, f)
    f.close()
    print('FINISHED SAVING PICKLE')


def load_pickle(pickle_path):
    f = open(pickle_path, 'rb')
    data = pickle.load(f)
    return data
