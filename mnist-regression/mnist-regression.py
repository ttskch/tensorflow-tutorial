from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# float値をとる784x可変長の2次元配列のプレースホルダ
# 訓練画像のピクセルごとのピクセル強度のデータ（784pix）を、訓練画像の枚数だけ持つためのもの
x = tf.placeholder(tf.float32, [None, 784])

# 784のfloat値が10個の入力ニューロンに入力される際のそれぞれの重み
# これから学習して決める値なので初期値はどうでもいいけど、とりあえず0パディングで初期化
W = tf.Variable(tf.zeros([784, 10]))

# 10個の出力ニューロンそれぞれのバイアス項
# これから学習して決める値なので初期値はどうでもいいけど、とりあえず0パディングで初期化
b = tf.Variable(tf.zeros([10]))

# NNモデル
# matmulはmatrix-multiple（行列の積）
y = tf.nn.softmax(tf.matmul(x, W) + b)
# チュートリアルには上記のコードが書いてあるけど、後の説明で交差エントロピーの計算にtf.nn.softmax_cross_entropy_with_logitsを使う場合は
# softmax関数はそこで適用されるみたいなので、ここでは下記でOK
# https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
y = tf.matmul(x, W) + b

# 訓練データの正解を持つためのプレースホルダ
y_ = tf.placeholder(tf.float32, [None, 10])

# 交差エントロピー
# reduce_meanは与えられた行列の要素の平均を計算する関数
# reduction_indices（axis）が1つだけの場合は戻り値はベクトル、それ以外はスカラー？
# https://www.tensorflow.org/api_docs/python/tf/reduce_mean
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1)) # これでも等価

# 実は上記の数式は数値的安定性が低いので、代わりにTensorFlow組み込みのsoftmax_cross_entropy_with_logitsを使う
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 学習率0.5で最急降下法によって交差エントロピーを最小化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# InteractiveSession上でモデルを起動する？
sess = tf.InteractiveSession()
tf.global_variables_initializer().run() # 作ったVariableをすべて初期化するというおまじないらしい

for _ in range(1000):
    # 訓練データから、画像とラベルの組をランダムに100件取得（確率的勾配降下法）
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # プレースホルダにバッチをフィードした状態でtrain_stepを実行
    # https://www.tensorflow.org/api_docs/python/tf/Session#run
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# y（出力結果）の1次元目（0〜9）とy_（正解）の1次元目（0〜9）が等しいかどうかを調べて、boolean値の配列を返す
# https://www.tensorflow.org/api_docs/python/tf/argmax
# https://www.tensorflow.org/api_docs/python/tf/equal
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# boolean配列をfloat32配列にキャスト（false=0,true=1）して、平均値を出す
# https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/cast
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close() # 作法的にはしたほうがいい？
