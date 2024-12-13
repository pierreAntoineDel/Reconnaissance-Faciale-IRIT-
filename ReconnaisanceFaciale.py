from sklearn.metrics import mean_squared_error
import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def print_eigenfaces(eigenV):
    fig = plt.fig=plt.figure(figsize=(16, 6))
    for i in range(23):
        ax=fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(eigenV[:, i].reshape(640, 480).T, cmap=plt.cm.bone)
    plt.show()

def restauration(eigenV,num_im,individu_moyen):
    y = []
    res = 0
    fig = plt.fig = plt.figure(figsize=(16, 6))
    for i in range(23):
        weight_i = np.dot(eigenV[:, i].transpose(), nmatX[num_im])
        res += np.dot(weight_i, eigenV[:, i])

        ax = plt.subplot(2, 15, i + 1)
        ax.set_title("k = " + str(i + 1))
        plt.imshow(res.reshape(640, 480).T + individu_moyen.reshape(640, 480).T, cmap='gray')

        rmse = np.sqrt(mean_squared_error(res, nmatX[num_im]))
        y.append(rmse)

    fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16)
    plt.show()

    plt.imshow(mat.__getitem__('X')[num_im].reshape(640, 480).T, cmap='gray')
    plt.show()
    plt.imshow(res.reshape(640, 480).T, cmap='gray')
    plt.show()

    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    plt.plot(x, y)
    plt.show()

# Exercice 1
mat = sc.loadmat('donnees.mat')
individi_moy = np.mean(mat.__getitem__('X'),0)
plt.imshow(individi_moy.reshape(640, 480).T, cmap='gray')
plt.title("Individu moyen")
plt.axis('off')
plt.show()

nmatX = mat.__getitem__('X')-individi_moy
trans_nmatX = nmatX.transpose()
cov_var = (nmatX.dot(trans_nmatX))/mat.__getitem__('n')
cov_var2 = (nmatX.dot(trans_nmatX))/mat.__getitem__('n')
valeurs_propres, vecteurs_propres=np.linalg.eig(cov_var2)
normalized_vecteurs_propres = vecteurs_propres/np.linalg.norm(vecteurs_propres)

# Exercice 2
idx = valeurs_propres.argsort()[::-1]
eigenValues = valeurs_propres[idx]
eigenVectors = normalized_vecteurs_propres[:,idx]
eigenVectors=nmatX.T@eigenVectors

print_eigenfaces(eigenVectors)

restauration(eigenVectors,12,individi_moy)

# Exercice 3
imgpil = Image.open("i00.png")
imgGray = imgpil.convert('L')
img = np.array(imgGray)

img=img.reshape(480, 640)-individi_moy.reshape(640, 480).T
img=img.reshape(307200)

fig = plt.fig = plt.figure(figsize=(16, 6))
y = []
res=0
for i in range(23):
    weight_i = np.dot(eigenVectors[:, i].T, img)
    res += individi_moy+np.dot(eigenVectors[:, i],weight_i)

    ax = plt.subplot(2, 15, i + 1)
    ax.set_title("k = " + str(i + 1))
    plt.imshow(res.reshape(640, 480).T, cmap='gray')

    rmse = np.sqrt(mean_squared_error(res, img))

    y.append(rmse)

fig.suptitle(("Reconstruction with Increasing Eigenfaces"), fontsize=16)
plt.show()

plt.imshow(img.reshape(480, 640),cmap='gray')
plt.show()
plt.imshow(res.reshape(640, 480).T,cmap='gray')
plt.show()

# Exercice 4-1 *

weight=[]
couleur = ['lightblue','red','black','yellow']
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
cc_num=0
for y in range (23):
    if(y==5 or y==11 or y==17):
        cc_num=cc_num+1
    for i in range(23):
        weight.append(np.dot(eigenVectors[:, i].T, nmatX[y]))
    plt.scatter(x, weight, c=couleur[cc_num])
    weight=[]


plt.title('Nuage de points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Exercice 4-2 V1*
for i in range(40):
    num_post = random.randint(1,6)
    num_ind = random.randint(1,15)
    if num_ind < 10:
        img_nom = 'i' + '0' + str(num_ind) + str(num_post)
    else:
        img_nom = 'i'+str(num_ind)+str(num_post)

    mat_test = sc.loadmat('Images/'+img_nom+'.mat')
    plt.imshow(mat_test.__getitem__(img_nom),cmap='gray')
    plt.show()
    img_test = mat_test.__getitem__(img_nom)


# Exercice 4-2 V2*
    img_test = img_test.transpose()
#imgpil = Image.open("i00.png")
#imgGray = imgpil.convert('L')
#img_test = np.array(imgGray).transpose()

    img_test = img_test-individi_moy.reshape(640, 480)
    img_test = img_test.reshape(307200)
    w = np.dot(img_test, eigenVectors)

    weights = np.dot(nmatX, eigenVectors)
    dist = np.min((weights - w) ** 2, axis=1)
    indiceImg = np.argmin(dist)
    mindist = np.sqrt(dist[indiceImg])

    print(mindist)

    seuil = 1e-10
    if mindist <= seuil:
        plt.imshow(nmatX[indiceImg].reshape(640, 480).T+individi_moy.reshape(640, 480).T,cmap='gray')
        plt.title("Individu trouvé")
        plt.show()
        print("individu trouvé")
    else:
        print("individu non trouvé")

# Final (cumul des exo)*
imgpil = Image.open("i00.png")
imgGray = imgpil.convert('L')
img_test = np.array(imgGray).transpose()
img_test = img_test-individi_moy.reshape(640, 480)
img_test = img_test.reshape(307200)
plt.imshow(img_test.reshape(640, 480).T+individi_moy.reshape(640, 480).T, cmap='gray')
plt.show()
weightssz = np.dot(eigenVectors.transpose(), img_test)
resultat=individi_moy

for i in range(23):
    resultat +=np.dot(weightssz[i], eigenVectors[:, i])

img_test = resultat
w = np.dot(img_test, eigenVectors)
weights = np.dot(nmatX, eigenVectors)
dist = np.min((weights - w) ** 2, axis=1)
indiceImg = np.argmin(dist)
mindist = np.sqrt(dist[indiceImg])

print(mindist)

seuil = 2
if mindist <= seuil:
    plt.imshow(mat.__getitem__('X')[indiceImg].reshape(640, 480).T,cmap='gray')
    plt.title("Individu trouvé")
    plt.show()
    print("individu trouvé")
else:
    print("individu non trouvé")
