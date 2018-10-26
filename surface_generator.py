import tensorflow as tf
import numpy as np
import open3d
import math


def generator(inputs, stddev):

    def grow(inputs, depth, max_depth):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True,
            fused=True
        )

        inputs = tf.nn.sigmoid(inputs)

        return inputs if depth == max_depth else [grow(inputs, depth + 1, max_depth) for _ in range(2)]

    def shrink(inputs_seq, depth, min_depth):

        inputs = tf.concat(inputs_seq, axis=1) if depth == min_depth else tf.concat(
            [shrink(inputs, depth - 1, min_depth) for inputs in inputs_seq], axis=1)

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True,
            fused=True
        )

        inputs = tf.nn.sigmoid(inputs)

        return inputs

    # inputs = shrink(grow(inputs, 0, 3), 3, 0)

    for _ in range(128):

        inputs = tf.layers.dense(
            inputs=inputs,
            units=32,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=stddev)
        )

        inputs = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            training=True,
            fused=True
        )

        inputs = tf.nn.sigmoid(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=4,
        use_bias=False,
        kernel_initializer=tf.random_normal_initializer(stddev=stddev)
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=True,
        fused=True
    )

    inputs = tf.nn.sigmoid(inputs)

    return inputs[:, :1], inputs[:, 1:]


inputs = tf.placeholder(tf.float32, [None, 3])
outputs = generator(inputs, stddev=0.03)

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    mesh_sphere = open3d.create_mesh_sphere(radius=1.0, resolution=1000)

    vertices = np.array(mesh_sphere.vertices)
    scale, colors = session.run(outputs, feed_dict={inputs: vertices})

    mesh_sphere.vertices = open3d.Vector3dVector(vertices * scale)
    mesh_sphere.vertex_colors = open3d.Vector3dVector(colors)

    # mesh_sphere.compute_vertex_normals()

    def call_back(visualizer):
        visualizer.get_render_option().background_color = np.asarray([0, 0, 0])
        visualizer.get_view_control().rotate(10.0, 0.0)
        return False

    open3d.draw_geometries_with_animation_callback([mesh_sphere], call_back)
