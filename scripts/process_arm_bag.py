import pickle
import sys

import numpy as np
import rosbag

def parse_twist(o):
    return (o.linear.x,
            o.linear.y,
            o.linear.z,
            o.angular.x,
            o.angular.y,
            o.angular.z)

def parse_twist_stamped(o):
    t = o.header.stamp.to_sec()
    v = parse_twist(o.twist)
    return t, v

def parse_odom(o):
    t = o.header.stamp.to_sec()
    v = parse_twist(o.twist.twist)
    return t, v

if __name__ == '__main__':
    """Process a bag file produced by the ABB arm into a pickle file.

    Output Keys
    -----------
    description : String describing source
    obs_times   : Timestamps of observations
    obs_values  : Body velocity observations in vector format
    true_times  : Timestamps of ground truth
    true_values : Ground truth body velocity observations in vector format
    """

    if len(sys.argv) < 3:
        print 'Please specify bag path and output path'
        exit(-1)

    bag_path = sys.argv[1]
    out_path = sys.argv[2]

    bag = rosbag.Bag(bag_path)
    out_file = open(out_path, 'wb')

    obs_times = []
    obs_vels = []
    true_times = []
    true_vels = []
    for topic, msg, t in bag.read_messages():

        if topic == '/odom_true':
            ts, v = parse_odom(msg)
            true_times.append(ts)
            true_vels.append(v)
        elif topic == '/velocity_raw':
            ts, v = parse_twist_stamped(msg)
            obs_times.append(ts)
            obs_vels.append(v)

    out = {'description' : 'ABB arm trial %s' % bag_path,
           'obs_times' : np.array(obs_times),
           'obs_values' : np.array(obs_vels),
           'true_times' : np.array(true_times),
           'true_values' : np.array(true_vels)}
    pickle.dump(out, out_file)
