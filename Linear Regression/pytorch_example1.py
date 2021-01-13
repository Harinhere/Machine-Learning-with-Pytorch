import torch
import torch.nn as nn
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import imageio


#Data creation
#Physical parameters
v=5.
gravity=9.8
#model parameters, i.e.,number of independent and
#dependent variables
input_size=1
output_size=1
#Prepare the physics based data
#number of samples for  training and test
nsamp=10
x_data=torch.zeros((nsamp,1))
y_data=torch.zeros((nsamp,1))


for i in range(0,nsamp):
    length=np.random.ranf()/2. 
    x_data[i]=length
    y_data[i]=4.*pi*pi*length/gravity
    


#Data normalization
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

x_data=normalize(x_data)
y_data=normalize(y_data)




class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    
    def forward(self,x):
        y_out=self.linear(x)
        return y_out
    
#Model description
model=LinearRegression(input_size,output_size)
#Mean squared error (MSE) loss function
criterion=nn.MSELoss()
#stochastic gradient descent (SGD) optimization
#lr is the learning-rate
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#Visualization of the learning-process
#create the image-array
image_seq=[]
fig,ax=plt.subplots(figsize=(8,8))


for epoch in range(2500):
    
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
    y_new=model(x_data).detach()
    if(epoch+1)%40==0:
        
        plt.cla()
        ax.scatter(x_data,y_data,label='Data',c='r')
        ax.plot(x_data,y_new,label='Fit',c='black')
        ax.set_xlabel('Length')
        ax.set_ylabel('Period*Period')
        ax.set_xlim(0., 1.1)
        ax.set_ylim(0., 1.1)
    
        ax.text(0.1, 0.9, 'epoch = %d' % epoch)
        ax.text(0.1, 0.8, 'Loss = %.4f' % loss.item())
    
        #Store the images in array
        fig.canvas.draw()       
        image=np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image=image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image_seq.append(image)
            

# check if the model is successful
#y_new=model(x_data).detach()
#
#
#plt.scatter(x_data,y_data,label='Data',c='r')
#plt.plot(x_data,y_new,label='Fit',c='black')
#plt.xlabel('Length')
#plt.ylabel('Period*Period')
#plt.legend()
#plt.savefig('pytorch_example1.jpeg',dpi=1200)

imageio.mimsave('pytorch_example1.gif', image_seq, fps=10)

        


