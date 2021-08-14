import os
import socket
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from scipy.interpolate import interp1d


def main():
    first_time = True
    with open("config.yaml") as f:
        d = yaml.safe_load(f)
        f.close()
    s_stats = socket.socket()
    s_stats.bind(('0.0.0.0', 5001))
    s_stats.listen(5)
    while True:
        print('ATTENDO NUOVA CONNESSIONE')
        stats_socket, stats_address = s_stats.accept()
        print('CONNESSIONE ACCETTATA, IN ATTESA DI RICEVERE IL PROSSIMO STREAM')
        try:
            rec = stats_socket.recv(4096).decode()
            filename, filesize = rec.split('<SEPARATOR>')
        except UnicodeDecodeError:
            print('UNICODE DECODE ERROR')
            del rec
            continue
        except ValueError:
            print('ERRORE SPLIT')
            continue
        print('INFORMAZIONI PRELIMINARI RICEVUTE')
        filename = os.path.basename(filename)
        filesize = int(filesize)
        print(filesize)
        if first_time:
            first_time = False
        total = len(rec)
        print('STREAM RICEVUTO, SCARICO IL .zip')
        with open(filename, "wb") as f:
            while True:
                bytes_read = stats_socket.recv(4096)
                total = total + len(bytes_read)
                print(total)
                f.write(bytes_read)
                if not bytes_read:
                    break
                if total >= filesize:
                    break
        print('FILE SCARICATO, ESTRAGGO')
        with zipfile.ZipFile(d['path'] + filename, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.getcwd() + '/' + filename.split('.')[0]))
        print('FILE SCOMPATTATO, INIZIO LA COMPUTAZIONE')
        frames = []
        trtbnb = [0] * 10000
        bnbtrt = [0] * 10000
        for file in os.listdir(os.path.join(os.getcwd() + '/' + filename.split('.')[0])):
            frames.append(pd.read_pickle(d['path'] + filename.split('.')[0] + '/' + file))
        result = pd.concat(frames).reset_index(drop=True)
        arr_a1 = result['TRT_bid'].to_numpy()
        arr_a2 = result['TRT_ask'].to_numpy()
        arr_b1 = result['BNB_ask'].to_numpy()
        arr_b2 = result['BNB_bid'].to_numpy()
        for line in range(result.shape[0]):
            bnbtrt[int(arr_a1[line] - arr_b1[line]) + 1000] += 1
            trtbnb[int(arr_b2[line] - arr_a2[line]) + 1000] += 1
        x = np.linspace(925, 1075, 150)
        plt.plot(x, bnbtrt[925:1075], marker='x', label='BNB>TRT')
        plt.plot(x, trtbnb[925:1075], marker='x', label='TRT>BNB')
        y = np.array(list(range(10000)))
        f1 = interp1d(y - 1000, bnbtrt)
        f2 = interp1d(y - 1000, trtbnb)
        start = min(max(bnbtrt), max(trtbnb))
        print(bnbtrt)
        print(trtbnb)
        # print(func(f1,start))
        # print(func(f2,f1(func(f1,start))))
        pair = {'s1': 0, 's2': 0, 'score': 0}
        for i in range(start, d['near'], -50):
            print(i)
            x1 = func1(f1, i,d)
            x2 = func1(f2, i,d)
            if x1 + x2 > 5 and x1 != -500 and x2 != -500:
                val = ((x1 + x2) * i / (1 + (100 / i)))
                if val > pair['score']:
                    pair['s1'] = round(x1, 1)
                    pair['s2'] = round(x2, 1)
                    pair['score'] = round((x1 + x2) * i, 1)
        print(pair)
        try:
            plt.hlines(pair['score'] / (pair['s1'] + pair['s2']), 1000 + pair['s2'], 1000 + pair['s1'], linestyles='dashed',
                   colors='k')
            plt.vlines(1000 + pair['s2'], 0, pair['score'] / (pair['s1'] + pair['s2']), colors='k')
            plt.vlines(1000 + pair['s1'], 0, pair['score'] / (pair['s1'] + pair['s2']), colors='k')
        except ZeroDivisionError:
            print('TROPPI POCHI DATI')
        image_name = int(time.time())
        plt.savefig(os.path.join(os.getcwd() + '/pictures/') + str(image_name) + '.png', format='png', dpi=300)
        plt.close()
        telegram(pair, d, image_name)


def func1(f, value,yml):
    n = -500
    x= np.linspace(-50, 50, 1000)
    for i in x:
        if is_near(int(f(i)), value,yml):
            if i > n:
                n = i
    return n


def telegram(pair, yml, imagename):
    url = 'https://api.telegram.org/bot' + yml['token']
    requests.post(url + '/sendPhoto', data={'chat_id':
                                                str(yml['app_id']),
                                            'caption':
                                                'PROSSIME SOGLIE ARBITRAGGIO VALUTATE SULLE PRECEDENTI ORE:\n'
                                                '<code>' + str(pair) + '</code>',
                                            'parse_mode':
                                                'HTML'
                                            },
                  files={'photo': open(os.path.join(os.getcwd() + '/pictures/') + str(imagename) + '.png', 'rb')})


def is_near(x, y,yml):
    if abs(x - y) < int(yml['near']):
        return True
    return False


if __name__ == '__main__':
    main()
