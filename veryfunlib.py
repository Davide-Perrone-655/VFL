import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64, MT19937



seed=10


def setseed(a):
	global seed
	seed=a
	print('seed: %d' %seed)



rg=Generator(PCG64(seed))


#bootstrap con varianza e media
def bootstrap(A, N=1, jump=0, est='mean', rep=3, R_print=False, *args, **kwargs):
	'''funzione per il bootstrap. Estimatore: varianza o media
	
	!!!! IMPORTANTE: BISOGNA AVER CARICATO UN GENERATORE DI NUMERI CASUALI !!!!!
	from numpy.random import Generator, PCG64, MT19937 
	
	
	A: vettore di dati 1-D
	N: larghezza dei bin = 2**N
	jump: numero di dati da saltare
	rep: numero di ripetizioni per bootstrap
	R_print: stampa il grafico dell'errore sull'estimatore per i valori di bin da 1 a N
	da implementare: altri estimatori
	'''

	#prima cosa: resize in modo che venga bene nella divisione per 2**N
	New=(len(A)-jump)
	Div=2**N
	if (New<=Div):
		d=int(np.log2(New/1000))
		print('N troppo grande, provare %d' %d)
		return
	L1=New - (New%Div)
	
	B=np.zeros(L1)
	C=np.zeros(L1)
	D=np.zeros(L1)
	Holder=np.zeros(rep)
	B[:]=A[ jump : (jump + L1)]
	
	if (N!=0):						#se ho chiesto di fare bootstrap correlato
		if (R_print==True):
			#print('R_print vero')
			Saved_data=np.zeros(N)
			if (est=='variance'):
				for i in range(N):
					inc=2**(i+1)
					L2=L1//inc					#riduco il mio array, lo preparo per il bootstrap
					C=np.resize(C, L2)
					D=np.resize(D, L2)

					for j in range(L2):				#medio i valori: definisco il mio vettore con bin
						C[j]=(sum(B[j:j+inc]))/inc
				
					for j in range(rep):
						for s in range(L2):
							k=int(rg.random()*L2)
							D[s] = C[k]			#riassegno a D
												
					
						Holder[j]=np.var(D,ddof=1)
					
					Temp1=np.sum(Holder)
					Temp2=np.dot(Holder,Holder)
					Saved_data[i]=np.sqrt((Temp2/rep) - ((Temp1/rep)**2))
					#print('terazione finita')
					
			elif(est=='mean'):
				#print('variabile: media')
				for i in range(N):
					inc=2**(i+1)
					L2=L1//inc					#riduco il mio array, lo preparo per il bootstrap
					C=np.resize(C, L2)
					D=np.resize(D, L2)

					for j in range(L2):				#medio i valori: definisco il mio vettore con bin
						C[j]=(sum(B[j:j+inc]))/inc
				
					for j in range(rep):
						for s in range(L2):
							k=int(rg.random()*L2)
							D[s] = C[k]			#riassegno a D
						Holder[j]=np.mean(D)
						
					Temp1=np.sum(Holder)
					Temp2=np.dot(Holder,Holder)
					Saved_data[i]=np.sqrt((Temp2/rep) - ((Temp1/rep)**2))
					#print('terazione finita')
			
			X=np.arange(0,N)
			plt.errorbar(X, Saved_data, markersize=5, linestyle='', marker='.')
			plt.title("Errore sull'estimatore dal bootstrap")
			plt.yscale('log')
			plt.ylabel("Sigma")
			plt.xlabel('Lunghezza bin, 2**N')
			plt.grid(color = 'silver')
			plt.show()
			res=Saved_data[N-1]
			
		else:
			#print('rprint falso')
			res=0
			inc=2**N
			L2=L1//inc					#riduco il mio array, lo preparo per il bootstrap
			C=np.resize(C, L2)
			D=np.resize(D, L2)
			for j in range(L2):				#medio i valori: definisco il mio vettore con bin
				C[j]=(sum(B[j:j+inc]))/inc
				
			if (est=='variance'):
				#print('varianza')
				for j in range(rep):
					for s in range(L2):
						k=int(rg.random()*L2)
						D[s] = C[k]			#riassegno a D
											
					Holder[j]=np.var(D,ddof=1)
				
				Temp1=np.sum(Holder)
				Temp2=np.dot(Holder,Holder)
				res=np.sqrt((Temp2/rep) - ((Temp1/rep)**2))
				#print('varianza fatta %f' %res)
			elif (est=='mean'):
				for j in range(rep):
					for s in range(L2):
						k=int(rg.random()*L2)
							
						D[s] = C[k]			#riassegno a D
										
					Holder[j]=np.mean(D)
					#print('giro fatto')
				Temp1=np.sum(Holder)
				Temp2=np.dot(Holder,Holder)
				res=np.sqrt((Temp2/rep) - ((Temp1/rep)**2))
		return res
		
		
		
		
	elif (N==0):
		C=np.resize(C, L1)						#riduco il mio array, lo preparo per il bootstrap
		D=np.resize(D, L1)
		if (est=='variance'):
			for j in range(rep):
				for s in range(L1):
					k=int(rg.random()*L1)
				
					D[s] = C[k]				#riassegno a D	
											
							#scelte di estimatori
				Holder[j]=np.var(D,ddof=1)
				
			res=np.std(Holder, ddof=1)
			
		elif (est=='mean'):
			for j in range(rep):
				for s in range(L1):
					k=int(rg.random()*L1)
				
					D[s] = C[k]				#riassegno a D		
											
				Holder[j]=np.mean(D)
				
			res=np.std(Holder, ddof=1)
		return	res
	else:
		res=0
		print('Qualcosa è andato storto')
		return res
		
		
		
	
#nuova funzione: data blocking		
def blocking(A, N=1, jump=0, R_print=False, *args, **kwargs):
	'''funzione di data blocking: calcola l'errore con blocchi di larghezza sempre maggiore, senza ripescare
	A: vettore di dati
	N: dimensione del blocco come 2**N
	jump: cose da saltare in caso di mancate termalizzazioni
	R_print: true per stampare il grafico con le varie iterazioni
	
	'''
	New=(len(A)-jump)
	Div=2**N
	
	if (New<=Div):							#controllo per evitare di tagliare in modo esagerato
		d=int(np.log2(New/1000))
		print('N troppo grande, provare %d' %d)
		return
	
	L1=New - (New%Div)
	B=np.zeros(L1)
	B[:]=A[ jump : (jump + L1)]
	
	mean1=np.mean(B)
	
	if (R_print==True):
		Saved_data=np.zeros(N)
		
		for i in range(N):
			var1=0
			inc=2**(i+1)
			L2=L1//inc					#riduco il mio array, lo preparo per il bootstrap
			
			for j in range(L2):				#medio i valori: definisco il mio vettore con bin
				temp1=(sum(B[j:j+inc]))/inc
				var1=var1 + (((temp1-mean1)**2)/L2)
				
			Saved_data[i]=np.sqrt(var1/(L2-1))
			
		X=np.arange(0,N)
		plt.errorbar(X, Saved_data, markersize=5, linestyle='', marker='.')
		plt.title("Errore da data blocking")
		plt.yscale('log')
		plt.ylabel("Errore")
		plt.xlabel('Lunghezza bin, 2**N')
		plt.grid(color = 'silver')
		plt.show()
		
		res=Saved_data[N-1]
		
	else:
		var1=0
		inc=2**N
		L2=L1//inc					#riduco il mio array, lo preparo per il bootstrap
		for j in range(L2):				#medio i valori: definisco il mio vettore con bin
			temp1=(sum(B[j:j+inc]))/inc
			var1=var1 + (((temp1-mean1)**2)/L2)
				
		res=np.sqrt(var1/(L2-1))
		
	return res
	
	
	
	
	
	
	
	
	
	
	
	
	
	



