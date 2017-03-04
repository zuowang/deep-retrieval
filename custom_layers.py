import caffe
import numpy as np


class NormalizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'
        self.eps = 1e-8

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data / np.expand_dims(self.eps + np.sqrt((bottom[0].data ** 2).sum(axis=1)), axis=1)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            tmp = np.expand_dims(self.eps + (top[0].data * top[0].diff ).sum(axis=1), axis=1)
            bottom[0].diff[:] = tmp * top[0].data
            bottom[0].diff[:] = top[0].diff - bottom[0].diff
            bottom[0].diff[:] = bottom[0].diff / np.expand_dims(self.eps + np.sqrt((bottom[0].data ** 2).sum(axis=1)), axis=1)
        #raise NotImplementedError("Backward pass not supported with this implementation")


class AggregateLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 1, 'This layer can only have one bottom'
        assert len(top) == 1, 'This layer can only have one top'

    def reshape(self, bottom, top):
        tmp_shape = list(bottom[0].data.shape)
        tmp_shape[0] = 1
        top[0].reshape(*tmp_shape)

    def forward(self, bottom, top):
        top[0].data[:] = bottom[0].data.sum(axis=0)

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            num = bottom[0].data.shape[0]
            for i in range(num):
                bottom[0].diff[i] = top[0].diff[0]
        #raise NotImplementedError("Backward pass not supported with this implementation")
