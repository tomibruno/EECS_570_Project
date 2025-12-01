import numpy as np

# min dim_in = 128 -> 256bit / 16bit
# min dim_out = 8 PIM block
BATCH = 1
REAL_DIM_IN = 1024
DIM_IN = 1024
DIM_OUT = 4096//2

np.set_printoptions(precision=20)
np.random.seed(1113)
data_type = 'float16'
batch_in = np.random.standard_normal(size=(DIM_IN, BATCH)).astype(data_type)
for i in range(REAL_DIM_IN, DIM_IN):
    for j in range(0, BATCH):
        batch_in[i][j] = 0

data_w = np.random.standard_normal(size=(DIM_OUT, DIM_IN)).astype(data_type)

np.random.shuffle(data_w)
batch_out = np.zeros((DIM_OUT, BATCH)).astype(data_type)
batch_out = np.matmul(data_w, batch_in)

batch_out2 = np.zeros((DIM_OUT, BATCH)).astype(data_type)

for y in range(0, DIM_OUT):
    for x in range(0, DIM_IN):
        # Convert float16 to uint16 bit representation
        w_bits = data_w[y][x].view(np.uint16)
        in_bits = batch_in[x][0].view(np.uint16)
        # Bitwise XOR and popcount
        xor_result = w_bits ^ in_bits
        popcount = bin(xor_result).count('1')
        batch_out2[y][0] += popcount
        # Print the original data values, w_bits, in_bits, and_Result and popcount for debugging
        #print(f"y={y}, x={x}, w={data_w[y][x]}, in={batch_in[x][0]}, w_bits={w_bits}, in_bits={in_bits}, and_result={and_result}, popcount={popcount}")

batch_in = batch_in.T.copy()
#batch_out = batch_out.T.copy()
batch_out = batch_out2.T.copy()
batch_out2 = batch_out2.T.copy()

np.save("hamming_input_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_in)
np.save("hamming_weight_" + str(DIM_OUT) + "x" + str(DIM_IN), data_w)
np.save("hamming_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out)
np.save("hamming_test_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out2)
print(batch_in)
print(batch_out)
print(batch_out2)
print(batch_in.shape)
print(batch_out.shape)
