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
from numpy.random import randn
from math import exp
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import CMA_EZ_CLASS as P

class MainWindow(qtw.QWidget):

    fcel = 0
    ogr = []
    lambda_ = 0
    Iter = 0
    Sigma = 0
    eps1 = 0
    eps2 = 0
    eps3 = 0
    n_ = 0
    muu = 0
    x_0 = []

    def __init__(self):
        super().__init__()

        # Nazwanie okna i określenie jego rozmiaru
        self.setWindowTitle("Algorytm CMA-ES")
        self.setMinimumSize(1200,800)
        self.label = qtw.QLabel()
        self.label.setText("Adaptacja macierzy kowariancji - strategia ewolucyjna")
        self.label.setFont(qtg.QFont('Arial',15))

        # Pola edytowalne
        self.Funkcja_celu = qtw.QLineEdit()
        self.Funkcja_celu.setText("Podaj funkcje celu...")
        self.Ograniczenia = qtw.QLineEdit()
        self.Ograniczenia.setText("Podaj ograniczenia...")
        self.Ilosc_sampli = qtw.QLineEdit()
        self.Ilosc_sampli.setText("30")
        self.Iteracje = qtw.QLineEdit();
        self.Iteracje.setText("150")
        self.Mu = qtw.QLineEdit()
        self.Mu.setText("5")
        self.Poczatkowy_krok = qtw.QLineEdit()
        self.Poczatkowy_krok.setText("0.2")
        self.Epsilon1 = qtw.QLineEdit()
        self.Epsilon1.setText("0")
        self.Epsilon2 = qtw.QLineEdit()
        self.Epsilon2.setText("0")
        self.Epsilon3 = qtw.QLineEdit()
        self.Epsilon3.setText("0")
        self.n = qtw.QLineEdit()
        self.n.setText("2")
        self.Punkt_pocz = qtw.QLineEdit()
        self.Punkt_pocz.setText("0,0")

        # Guziki
        self.Zapisz = qtw.QPushButton("Zapisz")
        self.Oblicz = qtw.QPushButton("Oblicz")
        self.Sprawdz = qtw.QPushButton("Sprawdź")
        self.F1 = qtw.QPushButton("(1-x[0])**2+100*(x[1]-x[0]**2)**2")
        self.F2 = qtw.QPushButton("4*x[0]**2-2.1*x[0]**4+0.333*x[0]**6+x[0]*x[1]-4*x[1]**2-1+x[1]**4")
        self.F3 = qtw.QPushButton("sin(x[1])*exp((1-cos(x[0]))**2)+cos(x[0])*exp((1-sin(x[1]))**2)+(x[0]-x[1])**2")
        self.Ogr1 = qtw.QPushButton("x[0]**2+x[1]**2 <= 2")
        self.Ogr2 = qtw.QPushButton("-sin(4*3.14*x[0])+2*sin(2*3.14*x[1])**2<=1.5")
        self.Ogr3 = qtw.QPushButton("(x[0]+5)**2+(x[1]+5)**2 < 25")

        # Wypisywanie informacji
        self.Output = qtw.QTextEdit()
        self.Output.setText("Konsola...")

        # Wykres
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Podział na sekcje
        self.sekcja_input()
        self.sekcja_guziki()

        # Utworzenie layoutu aplikacji
        mainLayout = qtw.QGridLayout()

        # Rysowanie elementów na pozycjach
        mainLayout.addWidget(self.input,1,0)
        mainLayout.addWidget(self.guziki,2,0)
        mainLayout.addWidget(self.canvas,1,2,2,1)
        mainLayout.addWidget(self.toolbar,0,2)
        mainLayout.addWidget(self.Output,4,0,1,4)
        mainLayout.addWidget(self.label,0,0,1,2)
        self.setLayout(mainLayout)

        # Podłączenie guzików
        self.Zapisz.clicked.connect(self.Guzik_Zapisz)
        self.Sprawdz.clicked.connect(self.Guzik_Sprawdz)
        self.F1.clicked.connect(self.Guzik_F1)
        self.F2.clicked.connect(self.Guzik_F2)
        self.F3.clicked.connect(self.Guzik_F3)
        self.Ogr1.clicked.connect(self.Guzik_Ogr1)
        self.Ogr2.clicked.connect(self.Guzik_Ogr2)
        self.Ogr3.clicked.connect(self.Guzik_Ogr3)
        self.Oblicz.clicked.connect(self.Guzik_Oblicz)

        self.show()

    def Guzik_Oblicz(self):
        self.figure.clear()
        #x = 0
        #XX = []
        #YY = []
        #opt = 0
        #ret_ogr = []
        #final_ogr = []
        try:
            x = P.cmaes(int(self.lambda_), float(self.Sigma), int(self.Iter), int(self.n_), self.fcel, self.ogr,
                        int(self.muu), float(self.eps1), float(self.eps2), float(self.eps3), self.x0)
            XX, YY, opt, ret_ogr, final_ogr = x.Algorytm
        except:
            print("Mógł wystąpić błąd parsowania, parametrów, itp..")
            print("Upewnij się, że wszystko jest ok i spróbuj ponownie")
            return

        for i in range(len(XX)):
            self.Output.append("Iteracja: %s" % (i))
            self.Output.append("Długość kroku: %s" % (x.Array_of_sigmas[i]))
            self.Output.append("X_1: %s" % (XX[i]))
            self.Output.append("X_2: %s" % (YY[i]))
        self.Output.append("Najlepsza średnia: ")
        self.Output.append("x*: %s" % (x.m))
        self.Output.append("f(x*): %s" % (opt))
        self.Output.append(x.flaga)

        self.Output.append("gi(x*)<0: ")
        for ogr in final_ogr:
            self.Output.append(ogr)

        if int(self.n_) == 2:
            print(self.n_)
            x.Draw(XX, YY, self.canvas)

    def Guzik_Zapisz(self):
        self.fcel = self.Funkcja_celu.text()
        self.ogr = self.Ograniczenia.text().split(',')
        self.lambda_ = self.Ilosc_sampli.text()
        self.Iter = self.Iteracje.text()
        self.Sigma = self.Poczatkowy_krok.text()
        self.muu = self.Mu.text()
        self.eps1 = self.Epsilon1.text()
        self.eps2 = self.Epsilon2.text()
        self.eps3 = self.Epsilon3.text()
        self.n_ = self.n.text()
        self.x0 = np.fromstring(self.Punkt_pocz.text(),dtype=float,sep=',')

    def Guzik_Sprawdz(self):
        self.Output.append("Funkcja celu: %s, Ograniczenia: %s, Ilość sampli populacji: %s, Ilość iteracji: %s, mu: %s, Epsilon1,2,3: %s,%s,%s, Ilość wymiarów problemu: %s, Punkt początkowy: %s" % (self.fcel,self.ogr,self.lambda_,self.Iter,self.muu,self.eps1,self.eps2,self.eps3,self.n_,self.x0))

    def Guzik_F1(self):
        self.fcel = "(1-x[0])**2+100*(x[1]-x[0]**2)**2"
        self.Funkcja_celu.setText("(1-x[0])**2+100*(x[1]-x[0]**2)**2")

    def Guzik_F2(self):
        self.fcel = "4*x[0]**2-2.1*x[0]**4+0.333*x[0]**6+x[0]*x[1]-4*x[1]**2-1+x[1]**4"
        self.Funkcja_celu.setText("4*x[0]**2-2.1*x[0]**4+0.333*x[0]**6+x[0]*x[1]-4*x[1]**2-1+x[1]**4")

    def Guzik_F3(self):
        self.fcel = "sin(x[1])*exp((1-cos(x[0]))**2)+cos(x[0])*exp((1-sin(x[1]))**2)+(x[0]-x[1])**2"
        self.Funkcja_celu.setText("sin(x[1])*exp((1-cos(x[0]))**2)+cos(x[0])*exp((1-sin(x[1]))**2)+(x[0]-x[1])**2")

    def Guzik_Ogr1(self):
        self.ogr = ["x[0]**2+x[1]**2 <= 2"]
        self.Ograniczenia.setText("x[0]**2+x[1]**2 <= 2")

    def Guzik_Ogr2(self):
        self.ogr = ["-sin(4*3.14*x[0])+2*sin(2*3.14*x[1])**2<=1.5"]
        self.Ograniczenia.setText("-sin(4*3.14*x[0])+2*sin(2*3.14*x[1])**2<=1.5")

    def Guzik_Ogr3(self):
        self.ogr = ["(x[0]+5)**2+(x[1]+5)**2 < 25"]
        self.Ograniczenia.setText("(x[0]+5)**2+(x[1]+5)**2 < 25")

    def sekcja_input(self):
        self.input = qtw.QGroupBox("Dane wejściowe")
        layout = qtw.QFormLayout()
        layout.addRow("Funkcja celu:", self.Funkcja_celu)
        layout.addRow("Ograniczenia:", self.Ograniczenia)
        layout.addRow("Lambda:", self.Ilosc_sampli)
        layout.addRow("Iteracje:", self.Iteracje)
        layout.addRow("Początkowy krok:", self.Poczatkowy_krok)
        layout.addRow("Epsilon1:", self.Epsilon1)
        layout.addRow("Epsilon2:", self.Epsilon2)
        layout.addRow("Epsilon3:", self.Epsilon3)
        layout.addRow("Mu:", self.Mu)
        layout.addRow("Ilość wymiarów n:",self.n)
        layout.addRow("Punkt początkowy x0:",self.Punkt_pocz)
        layout.addWidget(self.Zapisz)
        layout.addWidget(self.Oblicz)
        layout.addWidget(self.Sprawdz)
        self.input.setLayout(layout)

    def sekcja_guziki(self):
        self.guziki = qtw.QGroupBox("Przykładowe funkcje")
        layout = qtw.QFormLayout()
        layout.addRow("Funkcja Rosenbrocka:", self.F1)
        layout.addRow("Ograniczenia Rosenbrock:", self.Ogr1)
        layout.addRow("Funkcja Gomeza - Leviego:", self.F2)
        layout.addRow("Ograniczenia Gomeza - Leviego:", self.Ogr2)
        layout.addRow("Funkcja Mishry:", self.F3)
        layout.addRow("Ograniczenia Mishry:", self.Ogr3)
        self.guziki.setLayout(layout)

def main(args):
    app = qtw.QApplication([])
    app.setStyle('Windows')
    mw = MainWindow()
    app.exec_()

if __name__ == '__main__':
    main(sys.argv)
