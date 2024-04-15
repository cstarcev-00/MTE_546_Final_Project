import os 
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise

timestep = 0.033363
#file = 10
files = os.listdir('data')

kalman = KalmanFilter (dim_x=6, dim_z=3)

#state transition matrix
kalman.F = np.array([[1.,0.,0.,1.,0.,0.],
                     [0.,1.,0.,0.,1.,0.],
                     [0.,0.,1.,0.,0.,1.],
                     [0.,0.,0.,1.,0.,0.],
                     [0.,0.,0.,0.,1.,0.],
                     [0.,0.,0.,0.,0.,1.]])

#measurement matrix
kalman.H = np.array([[1.,0.,0.,0.,0.,0.],
                     [0.,0.,1.,0.,0.,0.],
                     [0.,0.,0.,0.,1.,0.]])

#uncertanty matrix
kalman.P *= 10

#measurement noise
kalman.R = np.eye(3)*5

#process nois
kalman.Q = Q_discrete_white_noise(dim=2, dt = timestep, var=0.1, block_size=3)

print(kalman.Q)

my_data = np.genfromtxt('data/'+files[3], delimiter=',')

for file in files:
    print('Using file: \n{}'.format(file))


    #print('data/'+files[file])
    my_data = np.genfromtxt('data/'+file, delimiter=',')

    time_ns = my_data[:,0]
    x_px = my_data[:,1]
    y_px = my_data[:,2]
    radius = my_data[:,3]
    depth = my_data[:,4]

    #print(my_data[:, 0:5])

    t = np.linspace(0, (time_ns[-1] - time_ns[0]), num=len(time_ns))

    #state matrix: x, xv, y, yv, z, zv
    kalman.x = np.array([x_px[0], 0, y_px[0], 0, depth[0], 0])

    
    #kalman_predictions = np.array((len(x_px), 1))
    kalman_predictions = []
    for idx, values in enumerate(zip(x_px, y_px, depth)):
        
        measurement = [values[0], values[1], values[2]]
        kalman.predict()
        kalman.update(measurement)

        #kalman_predictions[idx] = kalman.x
        kalman_predictions.append(kalman.x)

    x_pred = []
    y_pred = []
    z_pred = []
    x_vel = []
    y_vel = []
    z_vel = []
    for pred in kalman_predictions:
        x_pred.append(pred[0])
        y_pred.append(pred[2])
        z_pred.append(pred[4])
        x_vel.append(pred[1])
        y_vel.append(pred[3])
        z_vel.append(pred[5])


    fig, ax = plt.subplots(3)
    fig.suptitle('Measured position vs Kalman Prediction for:\n{}'.format(file))


    ax[0].plot(t, x_px)
    ax[0].plot(t, x_pred)
    ax[0].set_title("X Pixel Location Over Time")
    ax[0].legend(['Measured', 'Predicted'])

    ax[1].plot(t, y_px)
    ax[1].plot(t, y_pred)
    ax[1].set_title("Y Pixel Location Over Time")
    ax[1].legend(['Measured', 'Predicted'])

    ax[2].plot(t, depth)
    ax[2].plot(t, z_pred)
    ax[2].set_title("Depth Value Over Time")
    ax[2].legend(['Measured', 'Predicted'])

    '''
    ax[3].plot(t, x_vel)
    ax[3].set_title("X Pixel Velocity Over Time")
    
    ax[4].plot(t, y_vel)
    ax[4].set_title("Y Pixel Velocity Over Time")
    
    ax[5].plot(t, z_vel)
    ax[5].set_title("Depth Velocity Over Time")
    '''

    #print(kalman.P)

    plt.tight_layout()
    plt.show()
