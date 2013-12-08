#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np
import samples_triangle as st
import samples_carre as sc
import samples_hexagone as sh
import samples_octogone as so
from pylab import *


def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)
        #print "n :",n

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))
         #   print 'layers', self.layers
          #  print "i", i

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw


        
        # Return error
        #print '(error**2).sum()', (error**2).sum()
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    global ListeSamples
    ListeSamples =[]

    global titreGraphique
    titreGraphique =''
    
    def learn(network,samples, epochs=5000, lrate=.1, momentum=0.1):
        # Train
        print '    Train'
        erreur_ok_y,erreur_ok_x= [],[]# coordonnees des points qui sont ok
        erreur_ko_y,erreur_ko_x= [],[]# coordonnees des points qui sont Ko
        
        for i in range(epochs):
            n = np.random.randint(samples.size)
            #print 'n : ', n
            #print 'samples[input][n] : ',samples['input'][n]
            #print 'samples[output][n] : ',samples['output'][n]
            point = samples['input'][n]
            expectedOutput = samples['output'][n]
            output = network.propagate_forward(point )[0]
            nbLearn = 0
            while abs(output - expectedOutput) > 0.1:
                #print 'nbLearn = ',nbLearn
                nbLearn = nbLearn+1
                network.propagate_backward( expectedOutput, lrate, momentum )
                output = network.propagate_forward( point )[0]
        # Test
        print '    Test'
        for i in range(200):
            point =  ListeSamples[i]
            expectedOutput = 1.0
            if i > 100:
                expectedOutput = -1.0

            output = network.propagate_forward(point)[0]
##            print 'output  : ',output
           # print 'output type : ',type(output)
           # print 'expectedOutput type : ',type(expectedOutput)
          ##  print i, samples['input'][i], '%.2f' % o[0],            
         #   print '(expected %.2f)' % samples['output'][i]
            
            differenceEnValeurAbsolue = abs(output - expectedOutput)
         #   print 'differenceEnValeurAbsolue : ',differenceEnValeurAbsolue,' - abs(samples[output][i]) : ', abs(samples['output'][i]), '  - o[0]:', abs(o[0])
            if differenceEnValeurAbsolue <= 0.5:
                erreur_ok_y.append(output)
                erreur_ok_x.append(i)
            else :
                erreur_ko_y.append(output)
                erreur_ko_x.append(i)

        plot(erreur_ok_x, erreur_ok_y,'bx') # point correct bleu et en croix
        plot(erreur_ko_x, erreur_ko_y,'ro') # point  faux rouge et rond
        xlabel('Point')
        ylabel('Sortie')
        
        tailleListeCoordonnees = len(erreur_ok_x) +len(erreur_ko_x)
        titre = titreGraphique, " : Taux de reussite " , (len(erreur_ok_x)*100)/tailleListeCoordonnees, " %"
        title(titre)
        TitrePointOK = 'Points corrects : ', len(erreur_ok_x)
        TitrePointKO = 'Points incorrects : ', len(erreur_ko_x)
        legend((TitrePointOK, TitrePointKO ), 'best')
        axis([0,tailleListeCoordonnees,-2,2])
        grid()          
        show()
            
        
    ##############################################################################    
    nb_samples_triangle=3000                                                      #
    nb_samples_carre=3000                                                         #
    nb_samples_hexagone=3000                                                      #                                                  
    nb_samples_octogone=3000                                                     #
    ##############################################################################
    network = MLP(2,2,1)
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])
    samples_t = np.zeros(2*nb_samples_triangle, dtype=[('input',  float, 2), ('output', float, 1)])
    samples_c = np.zeros(2*nb_samples_carre, dtype=[('input',  float, 2), ('output', float, 1)])
    samples_h = np.zeros(2*nb_samples_hexagone, dtype=[('input',  float, 2), ('output', float, 1)])
    samples_o = np.zeros(2*nb_samples_octogone, dtype=[('input',  float, 2), ('output', float, 1)])
    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
##    print "Learning the OR logical function"
##    network.reset()
##    samples[0] = (0,0), 0    
##    samples[1] = (1,0), 1
##    samples[2] = (0,1), 1
##    samples[3] = (1,1), 1
##    
##    learn(network, samples)

    #Exemple 5: Learning triangulus
    # -------------------------------------------------------------------------    
    print "Learning the triangle"
    network.reset()
    network = MLP(2,3,1)
    triangle=st.samples(nb_samples_triangle)
    triangle.create_samples()
    for i in range (triangle.nb_samples):
        samples_t[i]=triangle.samples[i]
        
    ListeSamples = st.exercice.samples_list
    titreGraphique = 'Apprentissage du triangle'
    learn(network,samples_t)
##    #print "poids finaux du triangle", network.weights
##    #print "layer", network.layers
 
    #Exemple 6: Learning square
    # ---------------------------------------------------------------------
    print "Learning the carre"
    network.reset()
    network = MLP(2,4,1)
    carre=sc.samples(nb_samples_carre)
    carre.create_samples()
    for i in range (carre.nb_samples):
        samples_c[i]=carre.samples[i]
    ListeSamples = sc.exercice.samples_list
    titreGraphique = ' Apprentissage du carre'
    learn(network,samples_c)
    
    #print "layer", network.layers
    #print "poids finaux du carre", network.weights
    
##    #Exemple 6: Learning hexagone
    # ---------------------------------------------------------------------
    print "Learning the hexagone "
    network.reset()
    network = MLP(2,6,1)
    hexagone=sh.samples(nb_samples_hexagone)
    hexagone.create_samples()
    for i in range (hexagone.nb_samples):
        samples_h[i]=hexagone.samples[i]

    ListeSamples = sh.exercice.samples_list
    titreGraphique = 'Apprentissage de lHexagone'
    learn(network,samples_h)
    
##    #print "layer", network.layers
##    #print "poids finaux de hexagone", network.weights
##
##    #Exemple 6: Learning octogone
##    # ---------------------------------------------------------------------
    print "Learning the octogone"
    network.reset()
    network = MLP(2,8,1)
    octogone=so.samples(nb_samples_octogone)
    octogone.create_samples()
    for i in range (octogone.nb_samples):
        samples_o[i]=octogone.samples[i]

    ListeSamples = so.exercice.samples_list
    titreGraphique='Apprentissage de lOctogone'
    learn(network,samples_o)
    
##    #print "layer", network.layers
##    #print "poids finaux du carre", network.weights
    #---------------------------------------------------------------------
    #Affichage                                                           |
    #---------------------------------------------------------------------
    #triangle
##print 'Affichage triangle'
##x,y=[],[]
##for i in range(len(triangle.samples_in)):
##    x.append(triangle.samples_in[i][0][0])
##    y.append(triangle.samples_in[i][0][1])
##    plot(x, y, 'bx')
##
##x,y=[],[]
##
##for i in range(len(triangle.samples_out)):
##    x.append(triangle.samples_out[i][0][0])
##    y.append(triangle.samples_out[i][0][1])
##    plot(x, y,'rx') 
##
##   
##axis([-5,5,-5,5])
##grid()          
##show()
##
##
##    #---------------------------------------------------------------------
##    #Square
##print 'Affichage carre'
##
##x,y=[],[]
##for i in range(nb_samples_carre/2):
##    x.append(carre.samples_in[i][0][0])
##    y.append(carre.samples_in[i][0][1])
##    plot(x, y, 'bx')
##
##x,y=[],[]
##for i in range(nb_samples_carre/2):
##    x.append(carre.samples_out[i][0][0])
##    y.append(carre.samples_out[i][0][1])
##    plot(x, y,'rx') 
##
##   
##axis([-4,4,-4,4])
##grid()          
##show()
##
####    #---------------------------------------------------------------------
####    #hexagone
##print 'Affichage hexagone'
##x,y=[],[]
##for i in range(len(hexagone.samples_in)):
##    x.append(hexagone.samples_in[i][0][0])
##    y.append(hexagone.samples_in[i][0][1])
##    plot(x, y, 'bx')
##
##x,y=[],[]
##
##for i in range(len(hexagone.samples_out)):
##    x.append(hexagone.samples_out[i][0][0])
##    y.append(hexagone.samples_out[i][0][1])
##    plot(x, y,'rx') 
##
##   
##axis([-5,5,-5,5])
##grid()          
###show()
##    #---------------------------------------------------------------------
####    #octogone
##print 'Affichage Octogone'
##x,y=[],[]
##for i in range(len(octogone.samples_in)):
##    x.append(octogone.samples_in[i][0][0])
##    y.append(octogone.samples_in[i][0][1])
##    plot(x, y, 'bx')
##
##x,y=[],[]
##
##for i in range(len(octogone.samples_out)):
##    x.append(octogone.samples_out[i][0][0])
##    y.append(octogone.samples_out[i][0][1])
##    plot(x, y,'rx') 
##
##   
##axis([-5,5,-5,5])
##grid()          
##show()
