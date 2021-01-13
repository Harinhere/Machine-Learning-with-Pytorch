#Author:Harindranath Ambalampitiya, PhD (Theoretical physics)
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
from numpy import pi,sin,cos
import matplotlib.pyplot as plt
import imageio


#Data creation for non-linear regression
#Data is simulating a single slit diffraction pattern
#model parameters, i.e.,number of independent and
#dependent variables and hidden layers
input_size=1
output_size=1
n_hidden=256
#Prepare the physics based data
#number of samples for  training and test
nsamp=500
x_data=torch.zeros((nsamp,1))
y_data=torch.zeros((nsamp,1))
x_test_data=torch.zeros((nsamp,1))

for i in range(0,nsamp):
    phi=-4.*pi+8.*pi*np.random.ranf()
    x_data[i]=phi
    x_test_data[i]=-4.*pi+i*8.*pi/nsamp
    y_data[i]=(sin(phi/2.)/(phi/2.))**2
    #add some noise to y_data
    y_data[i]=y_data[i]-0.075+.15*np.random.ranf()
    


#Data normalization
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

x_data=normalize(x_data)
y_data=normalize(y_data)
x_test_data=normalize(x_test_data)



class NonLinearRegression(nn.Module):
    def __init__(self,input_size,n_hidden,output_size):
        super(NonLinearRegression,self).__init__()
        self.hidden=nn.Linear(input_size, n_hidden)
        self.predict=nn.Linear(n_hidden,output_size)
    
    def forward(self,x):
        y_out=Fun.relu(self.hidden(x))
        return self.predict(y_out)
    
#Model description
model=NonLinearRegression(input_size,n_hidden,output_size)
#Mean squared error (MSE) loss function
criterion=nn.MSELoss()
#stochastic gradient descent (SGD) optimization
#lr is the learning-rate
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

#Visualization of the learning-process
#create the image-array
image_seq=[]
fig,ax=plt.subplots(figsize=(8,8))


for epoch in range(5000):
    
    #initial prediction with a forward-pass
    y_predict=model(x_data)
    
    #compute the error functin
    loss=criterion(y_predict,y_data)
    #minimize error with gradients
    optimizer.zero_grad()
    #update the weights
    loss.backward()
    optimizer.step()
    
    #only for illustration purpose
    y_new=model(x_test_data).detach()
    if(epoch+1)%50==0:
        
        plt.cla()
        ax.scatter(x_data,y_data,label='Data',c='r')
        ax.plot(x_test_data,y_new,label='Fit',c='black')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('Intensity')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(0., 1.1)
    
        ax.text(-0.75, 0.9, 'epoch = %d' % epoch)
        ax.text(-0.75, 0.8, 'Loss = %.4f' % loss.item())
    
        #Store the images in array
        fig.canvas.draw()       
        image=np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image=image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image_seq.append(image)
            

# check if the model is successful
#y_new=model(x_test_data).detach()
#
#
#plt.scatter(x_data,y_data,label='Data',c='r')
#plt.plot(x_test_data,y_new,label='Fit',c='black')
#plt.xlabel(r'$\theta$')
#plt.ylabel('Intensity')
#plt.legend()
#plt.savefig('pytorch_example2.jpeg',dpi=1200)

imageio.mimsave('pytorch_example2.gif', image_seq, fps=10)

        


