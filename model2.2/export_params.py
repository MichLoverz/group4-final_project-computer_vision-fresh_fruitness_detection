import onnx
import numpy as np
import struct
import os

model = onnx.load('fruit_model.onnx')

params = {}
for node in model.graph.node:
    if node.op_type == 'Scaler':
        for attr in node.attribute:
            if attr.name == 'offset':
                params['scaler_mean'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'scale':
                params['scaler_scale'] = np.array(list(attr.floats), dtype=np.float32)
    elif node.op_type == 'SVMClassifier':
        for attr in node.attribute:
            if attr.name == 'coefficients':
                params['svm_coef'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'kernel_params':
                params['svm_kernel'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'support_vectors':
                params['svm_sv'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'vectors_per_class':
                params['svm_vpc'] = np.array(list(attr.ints), dtype=np.int32)
            elif attr.name == 'rho':
                params['svm_rho'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'prob_a':
                params['svm_prob_a'] = np.array(list(attr.floats), dtype=np.float32)
            elif attr.name == 'prob_b':
                params['svm_prob_b'] = np.array(list(attr.floats), dtype=np.float32)

n_sv = len(params['svm_coef'])
n_features = 547
print(f"Support vectors: {n_sv}")
print(f"SV data length: {len(params['svm_sv'])}")
print(f"Kernel params: {params['svm_kernel']}")
print(f"Rho: {params['svm_rho']}")
print(f"Vectors per class: {params['svm_vpc']}")
print(f"prob_a: {params['svm_prob_a']}, prob_b: {params['svm_prob_b']}")

out = '../fruit_freshness_app/assets/model_params.bin'
with open(out, 'wb') as f:
    f.write(struct.pack('<I', n_sv))
    f.write(struct.pack('<I', n_features))
    f.write(params['scaler_mean'].tobytes())
    f.write(params['scaler_scale'].tobytes())
    f.write(params['svm_coef'].tobytes())
    f.write(params['svm_sv'].tobytes())
    f.write(params['svm_rho'].tobytes())
    f.write(params['svm_kernel'][0:1].tobytes())  # gamma
    f.write(params['svm_prob_a'].tobytes())
    f.write(params['svm_prob_b'].tobytes())

print(f"Binary size: {os.path.getsize(out)/1024/1024:.1f} MB")

# Verify: run inference with both ONNX and manual to confirm match
import onnxruntime as ort
sess = ort.InferenceSession('fruit_model.onnx')
test_input = np.random.randn(1, 547).astype(np.float32)
onnx_out = sess.run(None, {'input': test_input})
onnx_label = onnx_out[0][0]

# Manual inference
scaled = (test_input[0] - params['scaler_mean']) * params['scaler_scale']
sv = params['svm_sv'].reshape(n_sv, n_features)
gamma = float(params['svm_kernel'][0])
coef = params['svm_coef']
rho = float(params['svm_rho'][0])

decision = 0.0
for i in range(n_sv):
    diff = scaled - sv[i]
    sq_dist = np.dot(diff, diff)
    k = np.exp(-gamma * sq_dist)
    decision += coef[i] * k
decision -= rho

manual_label = 1 if decision > 0 else 0
print(f"\nVerification:")
print(f"ONNX label: {onnx_label}")
print(f"Manual label: {manual_label}")
print(f"Decision value: {decision}")
print(f"Match: {onnx_label == manual_label}")

# Test with multiple random inputs
match_count = 0
total = 20
for _ in range(total):
    test_input = np.random.randn(1, 547).astype(np.float32)
    onnx_label = sess.run(None, {'input': test_input})[0][0]
    
    scaled = (test_input[0] - params['scaler_mean']) * params['scaler_scale']
    decision = 0.0
    for i in range(n_sv):
        diff = scaled - sv[i]
        sq_dist = np.dot(diff, diff)
        k = np.exp(-gamma * sq_dist)
        decision += coef[i] * k
    decision -= rho
    manual_label = 1 if decision > 0 else 0
    if onnx_label == manual_label:
        match_count += 1

print(f"\nBatch verification: {match_count}/{total} match")
