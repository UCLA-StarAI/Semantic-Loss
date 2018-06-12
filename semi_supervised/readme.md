This sub-repo contains code to reproduce semi-supervised learning experiments.

The semantic loss calculation happens from line 126 - 135 in semantic.py (with tf.name_scope('wmc')). Note in this code, the exactly-one constraint is directly encoded using the definition formula (Definition 1 in the "A semantic loss for deep learning with symbolic knowledge" paper). Please check out the supervised learning code to see how constraints can be compiled into a circuit that has a linear time complexity. In our supervised learning experiments, we use sentential decision diagram, abbreviated as SDD, as our target circuit representation.

Please provide the data_path , num_labeled, batch_size as arguments when running the code. For example if your mnist dataset is stored in the repo mnist_data and you want to run the code in the 1000-labled setting with a batch size of 32. Then you run semantic.py as the following
	python semantic.py --data_path mnist_data --num_labeled 1000 --batch_size 32
You may need to re-tune the batch size slightly to get the performance reported in the paper. However, even without any hyper-parameter tuning, you should observe a noticeable improvement compared with your base model for semi-supervised learning tasks.
Note the code is written in python 3.6. And we don't guarantee backward compatibility with python2. 

If you have further questions regarding our experiments, please don't hesitate to reach out to me at yliang@cs.ucla.edu.
