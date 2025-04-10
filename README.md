# H.264-INTRA-CODING
This a visual illustration on how H.264 intra coding chooses the modes and macro blocks based on the RDO
The Estimation of rate is a tricky thing. since, we don't have the rate information before CABAC or CAVLC.
We used a simple rate estimation before encoding by couting the no.of non zero values after the quantization.
The avarage time to generate the results is 17.8 seconds.
