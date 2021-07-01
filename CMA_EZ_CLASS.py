import math
import numpy as np
from scipy.interpolate import griddata
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.tri as tri
from scipy.linalg import sqrtm
from scipy.linalg import polar
from numpy import sin
from numpy import cos
from numpy import exp
from numpy import sqrt
from numpy.random import randn
from numpy import abs
from math import pi
from matplotlib.colors import LinearSegmentedColormap
import warnings
#warnings.filterwarnings("ignore")

class cmaes:
    def __init__(self, sample, krok, iteracje, wymiar, fcel, ogr, mu, eps1, eps2, eps3, x0):

        self.lambda_ = sample # ile sampli w nowej populacji
        self.sigma = krok # początkowy krok
        self.L = iteracje # max iter
        self.n = wymiar # wymiar problemu
        self.m =  x0 # punkt startowy, tzn średnia startowa
        self.beta = 0.05/(wymiar+2)
        self.fcel_ob = 10E40
        self.mu = mu
        self.fcel = self.parse(fcel)
        #self.fcel = fcel
        self.ograniczenia = ogr
        for i in range(0,len(self.ograniczenia)):
            self.ograniczenia[i] = self.parse(self.ograniczenia[i])
        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3
        self.Array_of_sigmas = [self.sigma]
        self.flaga = 'Kryterium iteracji'

        self.sciezka_izotropowa = np.zeros(self.n)
        self.sciezka_anizotropowa = np.zeros(self.n)
        self.fcel_ob = 10E19

       # wagi rozwiązań w populacji
        self.w_f = np.zeros(self.mu)
        for i in range(0, self.mu):
            self.w_f[i] = np.log(self.mu+0.5) - np.log(i+1)
        self.w_f = self.w_f/np.sum(self.w_f)

        self.mu_eff = np.sum(self.w_f)**2/np.sum(self.w_f**2)
        self.c1 = 2/((self.n+1.3)**2+self.mu_eff) # z literatury
        self.c_mu = min(1-self.c1, 2 * (self.mu_eff-2+1/self.mu_eff) / ((self.n+2)**2+self.mu_eff))

        self.c_anizo = 4/(self.n+4)
        self.c_izo = (self.mu_eff+2) / (self.n+self.mu_eff+5)

        self.chiN=self.n**0.5*(1-1/(4*self.n)+1/(21*self.n**2))# wartość oczekiwana rozkładu chi kwadrat (0,1)
        self.damps = 1.1 + 2*max(0, np.sqrt((self.mu_eff-1)/(self.n+1))-1) + self.c_izo

        self.c_cov = 0.6/(self.n**2+1)

        I = np.identity(self.n)
        self.C = I
        self.inv_sqrt_C = sqrtm(np.linalg.inv(self.C)) # odwrotność pierwiastka macierzy C: C^(-1/2)
        self.mp = self.m
        self.check = 0

    def parse_leq0(self):
        leqograniczenia = [None] * len(self.ograniczenia)
        for i in range(0,len(self.ograniczenia)):

            if self.ograniczenia[i].rfind("<=") != -1 :
                ssplit = self.ograniczenia[i].split("<=")
                leqograniczenia[i] = ssplit[0] + "-" + ssplit[1]
            elif self.ograniczenia[i].rfind("<") != -1 :
                ssplit = self.ograniczenia[i].split("<")
                leqograniczenia[i] = ssplit[0] + "-" + ssplit[1]
            elif self.ograniczenia[i].rfind(">=") != -1:
                ssplit = self.ograniczenia[i].split(">=")
                leqograniczenia[i] = "-( " + ssplit[0] + "-" + ssplit[1] + ")"
            elif self.ograniczenia[i].rfind(">") != -1:
                ssplit = self.ograniczenia[i].split(">")
                leqograniczenia[i] = "-( " + ssplit[0] + "-" + ssplit[1] + ") "
            elif self.ograniczenia[i].rfind("==") != -1:
                ssplit = self.ograniczenia[i].split("==")
                leqograniczenia[i] = ssplit[0] + "-" + ssplit[1]
        return leqograniczenia
    def parse(self,fcel):
        for i in range(1,self.n+1):
            fcel = fcel.replace("x"+str(i),"x["+str(i-1)+"]")
        fcel = fcel.replace("^","**")
        print(fcel)
        return fcel
    def Draw(self,XX,YY,canvas):
        plt.plot(XX[-1],YY[-1],'b+',markersize=12,zorder=5)
        plt.plot(XX,YY,'r.-',linewidth=0.5,zorder=4)

        NN = 15000
        ZZW = np.zeros(NN)
        #XY = rng.multivariate_normal(m,7*I,NN)
        XY  = 15*(np.random.rand(NN,2)-0.5) + self.m
        lim_max=XY.max()
        lim_min=XY.min()
        for i in range(len(XY)):
            x = XY[i]
            ZZW[i] = eval(self.fcel)
        #plt.plot(XY[:,0],XY[:,1])
        #plt.tricontour(XY[:,0],XY[:,1],ZZW,15, linewidths=0.5, colors='k')

        xi_ , yi_ = np.mgrid[lim_min:lim_max:500j, lim_min:lim_max:500j]
        zi_ = griddata(XY, ZZW, (xi_, yi_),method='cubic')

        cp = plt.contourf(xi_, yi_, zi_,zorder=1)
        plt.colorbar(cp)
        ct=plt.contour(xi_,yi_,zi_,colors='black', linestyles='dashed',zorder=2)
        plt.clabel(ct, inline=True, fontsize=10)
        ara_ara_z = 1
        for ogr in self.ograniczenia:
            ogr_parsowany = ogr
            #ogr_parsowany = ogr_parsowany.replace(">=","==")
            #ogr_parsowany = ogr_parsowany.replace("<=","==")
            #ogr_parsowany = ogr_parsowany.replace(">","==")
            #ogr_parsowany = ogr_parsowany.replace("<=","==")


            x = [xi_, yi_]
            arr_z = eval(ogr_parsowany).astype(int)
            ara_ara_z = ara_ara_z & arr_z
            #zz2 = griddata(XY, arr_z, (xi_, yi_),method='cubic')

            plt.contour(xi_, yi_, arr_z,1,colors='black',zorder=1)
        plt.contourf(xi_, yi_, ara_ara_z, 1, zorder=1, alpha=0.3)
    #plt.tricontour(XY[:,0],XY[:,1],y,15, linewidths=0.5, colors='k')
    #plt.plot(arr_yy[:,0],arr_yy[:,1],"b.")
    #im = plt.imshow( ( ), extent=(x[0].min(),x[0].max(),x[1].min(),x[1].max()),origin="lower", cmap="rb",zorder=3)

        canvas.draw()
        #plt.show()

    def Selekcjonuj(self,x_n,y_n,populacja):
        populacja = []
        # liczenie funkcji celu populacji
        for i in range(0,self.lambda_):
            x = np.array(x_n[i][0])
            wart = eval(self.fcel)
            pair = (y_n[i],wart) # zapisanie do struktuy pary dla łatwego sortowania
            populacja.append(pair) # zapamiętuje tylko y_i czyli rozrzut względem 0

        # sortowanie rozwiązań wg najlepszego dopasowania
        populacja = sorted(populacja, key=lambda x: x[-1])
        # selekcja \mu najlepszych rozwiązań rozwiązań
        populacja = populacja[:self.mu]


        sp,fp = zip(*populacja) # odzyskanie wyselekcjonowanych wyników i punktów
        y_w = np.zeros(self.n)
        y_wwT = np.zeros((self.n,self.n)) # do aktualizacji C rzędu mu
        self.mp = self.m # zapamiętanie poprzedniej średniej

        for i in range(0,len(sp)):
            y_w = y_w + sp[i] * self.w_f[i]
            y_wwT = y_wwT + self.w_f[i]*sp[i].reshape(-1,1)*sp[i]
        #print("Up_sr: +", y_w)
        self.m =  self.m + y_w


        return y_w, y_wwT

    def Aktualizuj_C_ze_sciezek(self,y_w,y_wwT):


        # poprawa liczenia ścieżki w warunkach skrajnych
        if (np.linalg.norm(self.sciezka_izotropowa) < 1.5*np.sqrt(self.n)):
            self.sciezka_anizotropowa = (1-self.c_anizo) * self.sciezka_anizotropowa + np.sqrt(1-(1-self.c_anizo)**2)*np.sqrt(self.mu_eff)*(self.m-self.mp)/self.sigma
        else:
            self.sciezka_anizotropowa = (1-self.c_anizo) * self.sciezka_anizotropowa

        self.C = (1 - self.c1 - self.c_mu) * self.C
        self.C = self.C + self.c1*self.sciezka_anizotropowa.reshape(-1,1)*self.sciezka_anizotropowa + self.c_mu *y_wwT

    def Spr_ogr(self,x_n):
        n_ograniczen_v = 0
        indeksy_v = []
        
        x = x_n[0]
        y_jest_zgodne = True
        for j in range(0,len(self.ograniczenia)):
            ogr = self.ograniczenia[j]
            if(not eval(ogr)):
                        # oflagowanie ile, i które ograniczenia zostały przekroczone
                indeksy_v.append(j)
                n_ograniczen_v = n_ograniczen_v + 1
                y_jest_zgodne = False
                
        return y_jest_zgodne, indeksy_v, n_ograniczen_v
        
        
    @property
    def Algorytm(self):

        # wektor ścieżki ograniczeń
        v = np.zeros((len(self.ograniczenia),self.n))
        w = np.zeros((len(self.ograniczenia),self.n))
        c_v = 0.6 # forgetting factor

        rng = default_rng()
        XX = [self.m[0]]
        YY = [self.m[1]]
        ZZ = []
        x = self.m
        yc = eval(self.fcel)
        ZZ.append(yc)
        t = 0
        populacja = []
        stop = False

        # jeżeli zdarzy się punkt startowy z poza obszaru dozwolonego
        help_factor = 10
        y_n = rng.multivariate_normal(np.zeros(self.n), help_factor * self.sigma * self.C, 1)
        x_n = self.m + y_n
        x0_gud,ind_p, n_p = self.Spr_ogr(np.array([self.m]))

        kontrola_przegrzania = 1000
        while  not x0_gud and kontrola_przegrzania > 0:
            y_n = rng.multivariate_normal(np.zeros(self.n), help_factor * self.sigma * self.C, 1)
            x_n = self.m + y_n
            x0_gud,ind_p, n_p = self.Spr_ogr(np.array(x_n))
            help_factor = 1.00001 * help_factor
            kontrola_przegrzania = kontrola_przegrzania - 1
        self.m = x_n
        if kontrola_przegrzania == 0:
            print("nie znaleziono poprawnego punktu startowego, podaj inny")


        while t < self.L:
            Xn = []
            Yn = []
            # generacja populacji
            p_bierz = 0
            kontrola_przegrzania = 200
            while p_bierz < self.lambda_ and kontrola_przegrzania > 0:
                y_n = rng.multivariate_normal(np.zeros(self.n),self.sigma * self.C,1)
                x_n = self.m + y_n

                w_ograniczeniach, ind_p, n_p = self.Spr_ogr(x_n)
                if w_ograniczeniach:
                    Xn.append(x_n)
                    Yn.append(y_n)
                    p_bierz = p_bierz + 1
                else:
                    suma = 0
                    for i in range(0,n_p):
                        # aktualizacja pamięci wektora przekroczeń
                        v[i] = (1 - c_v)*v[i] + c_v * y_n
                        # konstrukcja korekty
                        vvT = v[ind_p[i]].reshape(-1,1)*v[ind_p[i]]
                        suma = suma + vvT

                    self.C = self.C -  self.beta/(n_p) * suma
                kontrola_przegrzania = kontrola_przegrzania - 1
            if(kontrola_przegrzania <= 0):
                #raise NameError("IterLim")
                x = self.m[0]
                opt = eval(self.fcel)

                return XX, YY, opt
            #self.check = Yn
            #print(Yn)
            #print(Xn)
            Xn = np.array(Xn)
            #print(Xn.shape)
            y_w, y_wwT = self.Selekcjonuj(Xn,Yn,populacja)

            self.sciezka_izotropowa = (1-self.c_izo)* self.sciezka_izotropowa
            self.sciezka_izotropowa = self.sciezka_izotropowa + np.sqrt(1-(1-self.c_izo)**2)*np.sqrt(self.mu_eff)* np.matmul(self.inv_sqrt_C,(self.m-self.mp)[0]/self.sigma)
            self.Aktualizuj_C_ze_sciezek(y_w, y_wwT)
            self.inv_sqrt_C = sqrtm(np.linalg.pinv(self.C))
            
            d = (self.c_izo/self.damps)
            arg = np.linalg.norm(self.sciezka_izotropowa)/self.chiN
            self.sigma = self.sigma * np.exp(d*(arg - 1))
            self.Array_of_sigmas.append(self.sigma) # Może się przydać
                        
            XX.append(self.m[0][0])
            YY.append(self.m[0][1])
            x = self.m
            #print("============== t:" ,t ," ==============")
            #print("s: " , self.sigma)
            #print("m: " , x)
            t = t+1

            
        x = self.m[0]
        opt = eval(self.fcel)


        ret_ogr = self.parse_leq0()

        print(ret_ogr)
        final_ogr = []
        for i in range(0,len(self.ograniczenia)):
            final_ogr.append( ret_ogr[i] + " :  " + str(eval(ret_ogr[i])))
        #ret_ogr =0
        #final_ogr=0
        print(final_ogr)
        return XX, YY, opt, ret_ogr, final_ogr


# Rosenbrock
# fcel = "((1-x1)^2+100*(x2-x1^2)^2)"
# ograniczenia =["x[0]**2+x[1]**2 <= 2"]
# #ograniczenia=[]
# x0 =np.zeros(2)+[-1.1,-0.1]

# Bir[B]
#fcel ="sin(x[1])*exp((1-cos(x[0]))**2)+cos(x[0])*exp((1-sin(x[1]))**2) + (x[0]-x[1])**2"
#ograniczenia =["(x[0]+5)**2+(x[1]+5)**2 < 25","-10 <= x[0]","x[0]<= 0","x[1]<= 0","x[1]>= -6.50"]
#x0 =np.zeros(2)+[-8,-4]

# Gomez - Levi
# fcel = "4*(x[0]**2) - 2.1*(x[0]**4)+0.3333*(x[0]**6)+x[0]*x[1]-4*(x[1]**2) -1 + (x[1]**4)"
# ograniczenia = ["2*(sin(2*pi*x[1])**2)-sin(4*pi*x[0])<= 1.5 ","x[0] <=0.75","x[0] >= -1","x[1] >= -1","x[1] <=1"]
# x0 =np.zeros(2)+[-0.1,0.48]

#fcel= "(4-x[0])**2 + (4-x[1])**2"
#ograniczenia = []
#x0 = [-4,-3]

#fcel = "- (x[1] + 47)*sin(sqrt(abs(0.5*x[0]+(x[1] + 47)))) - x[0]*sin(abs(x[0]-x[1]-47))"
#ograniczenia = ["-512 <= x[0]" ,"x[0]<= 512","-512 <= x[1]","x[1] <= 512"]
#x0 =np.zeros(2)+[-7,80]

#fcel = "(x[0]-2)**2 + (x[1]-2)**2 +2*(x[0])*(x[1]) + 1.2*x[0]**4"
#ograniczenia = ["x[0]**2 + x[1]**2 <= 66**2"]
#x0 = [-0.99589213 ,-3.825367  ]

# CMA = cmaes(130, 1, 11, 2, fcel, ograniczenia, 30, 0, 0, 0, x0)
# XX,YY,opt = CMA.Algorytm()
# print("f(x): " , opt)
# ##CMA.Draw(XX,YY)
