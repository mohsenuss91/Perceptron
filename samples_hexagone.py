import random
#samples inside

#nb_samples=5
class samples:
    def __init__(self,nb_samples):
        self.samples_in=[]
        self.samples_out=[]
        self.nb_samples=nb_samples
        self.samples=[]
        self.samples_temp=[]
        self.samples_list=[]
        self.samples_temp_list=[]
        
    def create_samples(self):

##        print "Nombre d'exemple cree  pour l'hexagone: ",self.nb_samples
        #print "Points dans le hexagone : "
        while len(self.samples_in)<(self.nb_samples):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)
            if y>=-1 and  y<=1 and y<=x+2 and y<=-x+2 and y>=x-2 and y>=-x-2:    
                self.samples_in.append([x,y])
                self.samples_temp_list.append([x,y])
        
        for i in range (self.nb_samples):
            tuple_in = ((self.samples_in[i][0],self.samples_in[i][1]),1)
            self.samples_in[i]= tuple_in
            self.samples_list.append(tuple_in[0][0])
            
##        print "#---------------------------------------------------------#"
            
        #samples outside
        #print "Points HORS du hexagone : "
        while len(self.samples_out)<(self.nb_samples):
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)

            if y<-1 or  y>1 or y>x+2 or y>-x+2 or y<x-2 or y<-x-2:
                self.samples_out.append([x,y])
                self.samples_temp_list.append([x,y])
        
        for i in range (self.nb_samples):
            tuple_out = ((self.samples_out[i][0],self.samples_out[i][1]),0)
            self.samples_out[i]= tuple_out

        self.samples_temp = self.samples_in+self.samples_out
        for i in range(self.nb_samples):
            index=random.randint(0,len(self.samples_temp)-1)
            temp=self.samples_temp.pop(index)
            self.samples.append(temp)

        self.samples_list=self.samples_in+self.samples_out

        for i in range (len(self.samples_temp_list)):
            self.samples_list[i]=self.samples_temp_list[i]
            
##        print "#---------------------------------------------------------#"
exercice=samples(100)
exercice.create_samples()
##print 'Exercice dans lhexagone : '
##print exercice.samples
