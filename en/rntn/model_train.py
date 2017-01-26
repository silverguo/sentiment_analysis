from src import *
import configparser

def main():
    # read the configuration file
    config = configparser.ConfigParser()
    config.read('./config.ini')
    trainPath = config.get('DATA', 'STD_TRAIN')

    # class of rntn
    rntn = RecursiveNTensorN()
    rntn.model_train(trainPath)
    return

if __name__ == '__main__':
    # main function
    main()