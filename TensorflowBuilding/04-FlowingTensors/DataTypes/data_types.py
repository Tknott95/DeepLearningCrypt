# basic data type examples bromigo
# Chill and Master The Lego Blocks
# Basics = True Wisdom
# Simplicity - Modularity - Abstraction Thinking
import tensorflow as _tf

_c = _tf.constant(4.0, dtype=_tf.float64)
print("\n c = _tf.constant(4.0, dtype=tf.float64 \n")
print("\n c: ", _c)
print("\n _c.dtype: ", _c.dtype)

print("/n ____CASTING____")

f32 = _tf.constant([1,2,3], name='x', dtype=_tf.float32)

print("\n f32.dtype \n", f32.dtype)
print("\n RUNNING f32 = _tf.cast(f32, _tf.int64")

f32 = _tf.cast(f32,_tf.int64)
print("\n f32c.dtype: ",f32.dtype)



