#!/usr/bin/env python

# To do:
# Normalize non-categorical data
# Score against validation test
# Submission script

#Removes 'can't determine cores' message
NUM_CORES = 2

# Module/Library Imports
import json as js
import numpy as np
import pandas as pd
import tensorflow as tf
import time

# Read CSV Data Into Pandas
test_file = '../../test.csv'
train_file = '../../train.csv'
test_df = pd.read_csv(test_file)
full_train_df = pd.read_csv(train_file)

# Pull out desired number of training instances
num_inputs = 100000
rand_input_vec = np.random.choice(len(full_train_df.index), num_inputs, replace=False)
shortened_train_df = full_train_df.iloc[rand_input_vec,:]

# Separate into training and validation set
ratio_train = 0.75 # ratio_validate = 1 - ratio_train
num_train = int(round(ratio_train * num_inputs))
train_df = shortened_train_df.iloc[:num_train]
validate_df = shortened_train_df.iloc[num_train:]

# Parse datetime field
# Years go 2003-2015
datetime_features = train_df[['Dates']].values[:,0]
datetime_array = np.zeros((len(datetime_features), 13+12+31+1+1+7+1))

# Array layout: [13 year columns - 12 month columns - 31 day columns - 1 hour column - 1 min column - 7 week day columns - 1 year day column ] 

for i, date_entry in enumerate(datetime_features):
    dt = time.strptime(date_entry, "%Y-%m-%d %H:%M:%S")
    datetime_array[i, (dt.tm_year-2003)] = 1
    datetime_array[i, (dt.tm_mon+12)] = 1
    datetime_array[i, (dt.tm_mday+24)] = 1
    datetime_array[i, 56] = dt.tm_hour
    datetime_array[i, 57] = dt.tm_min
    datetime_array[i, (dt.tm_wday+58)] = dt.tm_min
    datetime_array[i, 65] = dt.tm_yday

# print np.sum(datetime_array, axis=0)
# print datetime_array.shape
# print np.isnan(datetime_array).any()
# exit()

# Create Data Numpy Arrays and Normalize
X_loc = train_df[['X']].values
X_loc_norm = (X_loc - np.mean(X_loc, axis=0)) / np.std(X_loc, axis=0)

Y_loc = train_df[['Y']].values
Y_loc_norm = (Y_loc - np.mean(Y_loc, axis=0)) / np.std(Y_loc, axis=0)

''' # Now getting day of week from datetime entry
dayOfWeek_dict = { 'Sunday': 0,
                   'Monday': 1,
                   'Tuesday': 2,
                   'Wednesday': 3,
                   'Thursday': 4,
                   'Friday': 5,
                   'Saturday': 6 }

dayOfWeek_features = train_df[['DayOfWeek']].values[:,0]
dayOfWeek_array = np.zeros((len(dayOfWeek_features),1))

for i, day in enumerate(dayOfWeek_features):
    dayOfWeek_array[i] = dayOfWeek_dict[day]
'''

# Police District (Catergorical)
PdDistricts_dict = { 'CENTRAL': 0,
                     'NORTHERN': 1,
                     'INGLESIDE': 2,
                     'PARK': 3,
                     'MISSION': 4,
                     'TENDERLOIN': 5,
                     'RICHMOND': 6,
                     'TARAVAL': 7,
                     'BAYVIEW': 8,
                     'SOUTHERN': 9 }

PdDistricts_features = train_df[['PdDistrict']].values[:,0]
PdDistricts_array =  np.zeros((len(PdDistricts_features),10))

for i, dist in enumerate(PdDistricts_features):
    PdDistricts_array[i, PdDistricts_dict[dist]] = 1 

# Create train data
X_train = np.hstack((datetime_array, PdDistricts_array, X_loc_norm, Y_loc_norm))
# X_train = np.hstack((X_loc_norm, Y_loc_norm))

# X_train = np.hstack((X_loc, Y_loc, PdDistricts_array, datetime_array))
# X_train_mean = np.mean(X_train, axis=0)
# X_train_std = np.std(X_train, axis=0)
# X_train_norm = (X_train - X_train_mean)/X_train_std
X_train_norm = X_train
# X_test = test_df[['X', 'Y']].values
# X_test_col_std = (X_test - X_train_mean)/X_train_std
Y = train_df[['Category']].values

# Create Enumerated Crime Dictionary
with open('../../crime_dict.json', 'r') as crime_dict:
    Y_dict = js.load(crime_dict)

# Create Outputs (Categorical)
Y_cat = np.zeros((len(Y), len(Y_dict)))
for i, entry in enumerate(Y):
    Y_cat[i][int(Y_dict[entry[0]])] = 1

# Meta-Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 50
display_step = 5

num_hidden = 40
n_features = X_train.shape[1]
n_classes = len(Y_dict)

# TF Graph Variables, Weights, and Biases
x = tf.placeholder("float", [None, n_features])
y = tf.placeholder("float", [None, n_classes])
W = tf.Variable(tf.random_normal([n_features,n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

pred = tf.sigmoid(tf.add(tf.matmul(x,W),b))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                intra_op_parallelism_threads=NUM_CORES)) as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(X_train)/batch_size)
        batch =  np.random.choice( len(X_train), batch_size, replace=False)
        # Loop over all batches
        for i in range(total_batch):
            sess.run(optimizer, feed_dict={x: X_train_norm[batch,:], y: Y_cat[batch,:]})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: X_train_norm[batch,:], y: Y_cat[batch,:]})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: X_train_norm, y: Y_cat})

