import argparse
import train
import predict

#docker run --publish 8000:8000 python-image --goal predict --data 'data/test_1.csv'
#docker run python-image --goal=predict --data='data/test_1.csv'
#docker build -t python-image .

parser = argparse.ArgumentParser(description = 'Parametry aplikacji')
parser.add_argument('--goal', type=str, help='Cel, jaki chcesz osiągnąć (train / predict / post-training)')
parser.add_argument('--data', type=str, help='Ścieżka do zbioru przeznaczonego to predykcji. Format : \'<ścieżka do pliku>\'')
parser.add_argument('--drift', type=str, help='Czy wygenerować dane do wykrycia dryfu (y/n)')
parser.add_argument('--dataset', type=str, help='Ścieżka do zbioru danych do treningu. Format : \'<ścieżka do pliku>\'')


args = parser.parse_args()
if args.goal == 'train' and args.drift in ['y', 'n'] and args.dataset is not None and args.dataset != '':
    drift = True if args.drift == 'y' else False
    train.train(args.dataset, drift)

if args.goal == 'post-training' and args.dataset != '' and args.dataset is not None:
    train.train(args.dataset, False, True)

if args.goal == 'predict' and args.data is not None and args.data != '':
    predict.predict(args.data)

