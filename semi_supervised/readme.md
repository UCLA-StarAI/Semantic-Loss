This sub-repo contains code to reproduce semi-supervised learning experiments.

The semantic loss calculation happens from line 126 - 135 in semantic.py (with tf.name_scope('wmc')). Note in this code, the exactly-one constraint is directly encoded using the definition formula (Definition 1 in the "A semantic loss for deep learning with symbolic knowledge" paper). Please check out the supervised learning code to see how constraints can be compiled into a circuit that has a linear time complexity. In our supervised learning experiments, we use sentential decision diagram, abbreviated as SDD, as our target circuit representation.

Please provide the data_path , num_labeled, batch_size as arguments when running the code. For example if your mnist dataset is stored in the repo mnist_data and you want to run the code in the 1000-labled setting with a batch size of 32. Then you run semantic.py as the following
	python semantic.py --data_path mnist_data --num_labeled 1000 --batch_size 32

You may need to re-tune hyperparameters slightly to get the performance reported in the paper. However, even without any hyper-parameter tuning, you should observe a significant improvement compared with your base model for semi-supervised learning tasks.

Hyper-parameters' tunning ranges are as follows
	batch size: [10, 16, 32, 64, 128, 256, 512]
	standard deviation of guassian noise: [0.1, 0.2, 0.3, 0.4, 0.5]
	weight of semantic loss: [1.0, 0.5, 0,25, 0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
	learning rate of Adam optimizer: [1e-5, 1e-4, 1e-3]
	alpha of leaky relu: [0.01, 0.05]
In our experience, different values of the last two hyper-parameters do not cause noticeable difference in our model's prediction accuracy. So if you eventually decided to do parameter tuning over our model, you can start with focusing on the first three while using the default values in the code for the last two. It recently occured to us that the results on the setting with 100 labeled samples are very volatile; they are even afffected by the system environment, tensorflow version, etc.. We would suggest paying more attention to our results on the settings with more labeled samples, which are more stable and easier to reproduce.

We understand it would be very time consuming to do a full round of parameter tuning, especially on the cifar-10 dataset. To faciliate replication, we created a google drive repo that stores the exact model, input data and hyper-parameter settings we used when we ran the cifar-10 experiments. Please use the following link to download it: https://drive.google.com/drive/u/0/folders/128gYP3bHu7pvdlY-qbrS0UmO43Azjvmj

Note the code is written in python 3.6. And we don't guarantee backward compatibility with python2. 

If you have further questions regarding our experiments, please don't hesitate to reach out to me at yliang@cs.ucla.edu.
