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

        print "Nombre d'exemple cree pour le triangle : ",self.nb_samples
        #print "Points dans le triangle : "
        while (len(self.samples_in)+len(self.samples_out))<(self.nb_samples):
            #Generation des coordonnees x et y de maniere aleatoire
         
            x = random.uniform(-5,5)
            y = random.uniform(-5,5)

            #si les coordonnees verifie les equations des droites
            #qui composent le triangle ou non
            if y>=0 and y<=x+2 and y<=-x+2:
                #Point DANS le triangle
                self.samples_in.append([x,y])
                self.samples_temp_list.append([x,y])
            else :
                #Point HORS le triangle
                self.samples_out.append([x,y])
                self.samples_temp_list.append([x,y])

        print 'Nombre dexemples in : ',len(self.samples_in)
        print 'Nombre dexemples out : ',len(self.samples_out)
        # pour le nombre dexemple, on va creer les listes de tuples in
        # et de tuples out
        # un tuple = coordonnes x , coordonnes y, resultat attendu
        for i in range (len(self.samples_in)):
            
            tuple_in = ((self.samples_in[i][0],self.samples_in[i][1]),1)
            self.samples_in[i]= tuple_in
            self.samples_list.append(tuple_in[0][0])

            tuple_out = ((self.samples_out[i][0],self.samples_out[i][1]),0)
            self.samples_out[i]= tuple_out     
        
        
        self.samples_temp=self.samples_in+self.samples_out
        for i in range(self.nb_samples):
            index=random.randint(0,len(self.samples_temp)-1)
            temp=self.samples_temp.pop(index)
            self.samples.append(temp)

        self.samples_list=self.samples_in+self.samples_out

        for i in range (len(self.samples_temp_list)):
            self.samples_list[i]=self.samples_temp_list[i]

        print "#---------------------------------------------------------#"
exercice=samples(100)
exercice.create_samples()


