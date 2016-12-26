from kaffe_network import Network

class JSTL(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 32, 1, 1, name='conv1')
             .conv(3, 3, 32, 1, 1, name='conv2')
             .conv(3, 3, 32, 1, 1, name='conv3')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(1, 1, 64, 1, 1, name='inception_1a_1x1'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, name='inception_1a_3x3_reduce')
             .conv(3, 3, 64, 1, 1, name='inception_1a_3x3'))

        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, name='inception_1a_double_3x3_reduce')
             .conv(3, 3, 64, 1, 1, name='inception_1a_double_3x3_1')
             .conv(3, 3, 64, 1, 1, name='inception_1a_double_3x3_2'))

        (self.feed('pool1')
             .avg_pool(3, 3, 1, 1, name='inception_1a_pool')
             .conv(1, 1, 64, 1, 1, name='inception_1a_pool_proj'))

        (self.feed('inception_1a_1x1',
                   'inception_1a_3x3',
                   'inception_1a_double_3x3_2',
                   'inception_1a_pool_proj')
             .concat(3, name='inception_1a_output')
             .conv(1, 1, 64, 1, 1, name='inception_1b_3x3_reduce')
             .conv(3, 3, 64, 2, 2, name='inception_1b_3x3'))

        (self.feed('inception_1a_output')
             .conv(1, 1, 64, 1, 1, name='inception_1b_double_3x3_reduce')
             .conv(3, 3, 64, 1, 1, name='inception_1b_double_3x3_1')
             .conv(3, 3, 64, 2, 2, name='inception_1b_double_3x3_2'))

        (self.feed('inception_1a_output')
             .max_pool(3, 3, 2, 2, name='inception_1b_pool'))

        (self.feed('inception_1b_pool',
                   'inception_1b_3x3',
                   'inception_1b_double_3x3_2')
             .concat(3, name='inception_1b_output')
             .conv(1, 1, 128, 1, 1, name='inception_2a_1x1'))

        (self.feed('inception_1b_output')
             .conv(1, 1, 128, 1, 1, name='inception_2a_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_2a_3x3'))

        (self.feed('inception_1b_output')
             .conv(1, 1, 128, 1, 1, name='inception_2a_double_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_2a_double_3x3_1')
             .conv(3, 3, 128, 1, 1, name='inception_2a_double_3x3_2'))

        (self.feed('inception_1b_output')
             .avg_pool(3, 3, 1, 1, name='inception_2a_pool')
             .conv(1, 1, 128, 1, 1, name='inception_2a_pool_proj'))

        (self.feed('inception_2a_1x1',
                   'inception_2a_3x3',
                   'inception_2a_double_3x3_2',
                   'inception_2a_pool_proj')
             .concat(3, name='inception_2a_output')
             .conv(1, 1, 128, 1, 1, name='inception_2b_3x3_reduce')
             .conv(3, 3, 128, 2, 2, name='inception_2b_3x3'))

        (self.feed('inception_2a_output')
             .conv(1, 1, 128, 1, 1, name='inception_2b_double_3x3_reduce')
             .conv(3, 3, 128, 1, 1, name='inception_2b_double_3x3_1')
             .conv(3, 3, 128, 2, 2, name='inception_2b_double_3x3_2'))

        (self.feed('inception_2a_output')
             .max_pool(3, 3, 2, 2, name='inception_2b_pool'))

        (self.feed('inception_2b_pool',
                   'inception_2b_3x3',
                   'inception_2b_double_3x3_2')
             .concat(3, name='inception_2b_output')
             .conv(1, 1, 256, 1, 1, name='inception_3a_1x1'))

        (self.feed('inception_2b_output')
             .conv(1, 1, 256, 1, 1, name='inception_3a_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_3a_3x3'))

        (self.feed('inception_2b_output')
             .conv(1, 1, 256, 1, 1, name='inception_3a_double_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_3a_double_3x3_1')
             .conv(3, 3, 256, 1, 1, name='inception_3a_double_3x3_2'))

        (self.feed('inception_2b_output')
             .avg_pool(3, 3, 1, 1, name='inception_3a_pool')
             .conv(1, 1, 256, 1, 1, name='inception_3a_pool_proj'))

        (self.feed('inception_3a_1x1',
                   'inception_3a_3x3',
                   'inception_3a_double_3x3_2',
                   'inception_3a_pool_proj')
             .concat(3, name='inception_3a_output')
             .conv(1, 1, 256, 1, 1, name='inception_3b_3x3_reduce')
             .conv(3, 3, 256, 2, 2, name='inception_3b_3x3'))

        (self.feed('inception_3a_output')
             .conv(1, 1, 256, 1, 1, name='inception_3b_double_3x3_reduce')
             .conv(3, 3, 256, 1, 1, name='inception_3b_double_3x3_1')
             .conv(3, 3, 256, 2, 2, name='inception_3b_double_3x3_2'))

        (self.feed('inception_3a_output')
             .max_pool(3, 3, 2, 2, padding='SAME', name='inception_3b_pool'))

        (self.feed('inception_3b_pool',
                   'inception_3b_3x3',
                   'inception_3b_double_3x3_2')
             .concat(3, name='inception_3b_output')
             .avg_pool(9, 4, 1, 1, padding='VALID', name='global_pool')
             .fc(256, relu=False, name='fc7'))
