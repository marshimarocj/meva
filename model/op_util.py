import tensorflow as tf

def sigmoid_focal_loss(labels, logits, gamma=2.):
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  invprobs = tf.log_sigmoid(-logits * (labels * 2 - 1))
  w = tf.stop_gradient(tf.exp(invprobs * gamma))
  loss = w * loss

  return loss
