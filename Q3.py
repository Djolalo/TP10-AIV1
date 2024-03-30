import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



ref = "./Fichiers_utiles_TP10_2023/IMAGE3D_V16.bmp"



def seuilOpti(img, parf= None):
    init = np.array(img)

    histo = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(int)
    plt.plot(histo)
    plt.show()
    seuil_optimal = 0
    prob_max = -np.inf
    total = img.size
    for t1 in range(256):
        N1 = np.sum(histo[:t1])
        P1 = N1 / total
        mu1 = 0 if N1 == 0 else np.sum(np.arange(0,t1) * histo[0: t1]) / N1 # calcul de la moyenne 

        P2 = 1 - P1
        N2 = total - N1
        mu2 = 0 if N2 == 0 else np.sum(np.arange(t1+1, 256) * histo[t1+1:]) / N2
        #print(f'N1 = {N1}\t N2 = {N2}\t noccu1  = {noccu1}\t noccu2 = {noccu2}\t mu1 = {mu1}\t mu2 = {mu2}')
        prob_courante = P1 * P2 * (mu1-mu2)**2 
        if prob_courante > prob_max:
            seuil_optimal = t1
            prob_max = prob_courante
    if parf is not None: 
        _, img_seuil_1 = cv2.threshold(img, seuil_optimal[0], 127, cv2.THRESH_BINARY)
        _, img_seuil_2 = cv2.threshold(img, seuil_optimal[1], 255, cv2.THRESH_BINARY)
        dest = np.maximum(img_seuil_1, img_seuil_2)
        GT_1D = parf.flatten()
        dest_1D = dest.flatten()
        cm = confusion_matrix(GT_1D, dest_1D)
        accuracy = (np.diag(cm).sum() / img.size) * 100  # Pour avoir en %
        print(f"Taux de bonne classification des 3 classes: {accuracy}%")
    return seuil_optimal

if __name__ == "__main__":
    chans = cv2.split(cv2.imread(ref,  cv2.IMREAD_ANYCOLOR))
    threshold_values = [seuilOpti(chan) for chan in chans]
    chans = [cv2.threshold(chan, thresh, 255, cv2.THRESH_BINARY)[1] for chan, thresh in zip(chans, threshold_values)]
    final = cv2.merge(chans)
    cv2.namedWindow("Image finale", cv2.WINDOW_NORMAL)
    cv2.imshow("Image finale", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
