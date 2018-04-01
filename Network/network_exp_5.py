import tensorflow as tf
import Transform_Data as td
import numpy as np

max_accuracy=0
timer=0
confusion_matrics=np.zeros([7,7],dtype="int")
learning_rate=1e-4

print("Processing Training Data ...")
td.prepare_training_data()
print("Training Data Completed !")

print("Processing Validation Data ...")
validation=td.prepare_validation_data(100)
print("Validation Data Completed !")

print("Processing Testing Data ...")
test=td.prepare_testdata(100)
print("Test Data Completed !")

sess=tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('Input'):
    with tf.name_scope('Input_x'):
        x = tf.placeholder(tf.float32,shape=[None,1024])
    with tf.name_scope('Input_y'):
        y_ = tf.placeholder(tf.float32,shape=[None,7])

with tf.name_scope('Conv_1'):
    with tf.name_scope('W_conv1'):
        w_conv1=weight_variable([3,3,1,8])
    with tf.name_scope('B_conv1'):
        b_conv1=bias_variable([8])
    with tf.name_scope('x_image'):
        x_image=tf.reshape(x,[-1,32,32,1])
    with tf.name_scope('H_conv1'):
        h_conv1=tf.nn.sigmoid(conv2d(x_image,w_conv1)+b_conv1)

with tf.name_scope('Conv_2'):
    with tf.name_scope('W_conv2'):
        w_conv2=weight_variable([3,3,8,16])
    with tf.name_scope('B_conv2'):
        b_conv2=bias_variable([16])
    with tf.name_scope('H_conv2'):
        h_conv2=tf.nn.sigmoid(conv2d(h_conv1,w_conv2)+b_conv2)
    with tf.name_scope('H_pool2'):
        h_pool2=max_pool_2x2(h_conv2)

with tf.name_scope('Conv_3'):
    with tf.name_scope('W_conv3'):
        w_conv3=weight_variable([3,3,16,32])
    with tf.name_scope('B_conv3'):
        b_conv3=bias_variable([32])
    with tf.name_scope('H_conv3'):
        h_conv3=tf.nn.sigmoid(conv2d(h_pool2,w_conv3)+b_conv3)

with tf.name_scope('Conv_4'):
    with tf.name_scope('W_conv4'):
        w_conv4=weight_variable([3,3,32,64])
    with tf.name_scope('B_conv4'):
        b_conv4=bias_variable([64])
    with tf.name_scope('H_conv4'):
        h_conv4=tf.nn.sigmoid(conv2d(h_conv3,w_conv4)+b_conv4)
    with tf.name_scope('H_pool4'):
        h_pool4=max_pool_2x2(h_conv4)

with tf.name_scope('Full_Connected_Layer_1'):
    with tf.name_scope('W_fc1'):
        w_fc1=weight_variable([8*8*64,1024])
    with tf.name_scope('B_fc1'):
        b_fc1=bias_variable([1024])
    with tf.name_scope('H_pool_flat'):
        h_pool_flat=tf.reshape(h_pool4,[-1,8*8*64])
    with tf.name_scope('H_fc1'):
        h_fc1=tf.nn.relu(tf.matmul(h_pool_flat,w_fc1)+b_fc1)

with tf.name_scope('Full_Connected_Layer_2'):
    with tf.name_scope('W_fc2'):
        w_fc2=weight_variable([1024,7])
    with tf.name_scope('B_fc2'):
        b_fc2=bias_variable([7])
    with tf.name_scope('Y_conv'):
        y_conv=tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)

with tf.name_scope('Cross_Entropy'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    tf.summary.scalar('Cross_Entropy',cross_entropy)
with tf.name_scope('Train_Step'):
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('Correct_prediction'):
    distribution=[tf.arg_max(y_,1),tf.argmax(y_conv,1)]
    correct_prediction=tf.equal(distribution[0],distribution[1])
with tf.name_scope('Accuracy'):
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    tf.summary.scalar('Accuracy',accuracy)

sess.run(tf.global_variables_initializer())
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('D:/Log',sess.graph)

epoch=0
validation_accuracy_set=[]
avg_validation_accuracy=0
for i in range(407101): #300个Epoch Origin
    batch = td.next_batch(40)
    # print(batch[0])
    # print(batch[1])
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 1357 ==0:
        epoch += 1
        train_accuracy=accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        for image,label in zip(validation[0],validation[1]):
            validation_accuracy=accuracy.eval(feed_dict={x:image,y_:label})
            validation_accuracy_set.append(validation_accuracy)
        for item in validation_accuracy_set:
            avg_validation_accuracy += item
        validation_accuracy=avg_validation_accuracy/len(validation_accuracy_set)
        if validation_accuracy<0.8:
            learning_rate=1e-4
        if validation_accuracy>=0.8:
            learning_rate=5e-5
        if validation_accuracy>=0.9:
            learning_rate=1e-5
        validation_accuracy_set=[]
        avg_validation_accuracy=0
        print("Epoch %d , training accuracy %g,Validation Accuracy: %g" % (epoch, train_accuracy, validation_accuracy))
        result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1]})
        writer.add_summary(result, epoch)
        if validation_accuracy>max_accuracy:
            max_accuracy=validation_accuracy
            timer=0
        else:
            timer+=1
            if timer>10:
                break

for image,label in zip(test[0],test[1]):
    matrix_row,matrix_col=sess.run(distribution,feed_dict={x: image,y_:label})
    # print(matrix_row)
    # print(matrix_col)
    # print()
    for i,j in zip(matrix_row,matrix_col):
        confusion_matrics[i][j]+=1

test_accuracy_set=[]
avg_test_accuracy=0
for image,label in zip(test[0],test[1]):
    test_accuracy = sess.run(accuracy, feed_dict={x: image, y_: label})
    test_accuracy_set.append(test_accuracy)
for item in test_accuracy_set:
    avg_test_accuracy+=item
print("Average Test Accuracy :",avg_test_accuracy/len(test_accuracy_set))
print(np.array(confusion_matrics.tolist()))