import pandas as pd
import pickle

pkl_file_path = './121113.pkl'
file_path = '/home/ssu35/lion/final/sorted_paragraphs.csv'

with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

    

word_vecs = params['word_vecs']
word_to_id = params['word_to_id']
id_to_word = params['id_to_word'] 



df = pd.read_csv(file_path)
paragraphs = df['paragraphs'].values
scores = df['score'].values


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


class MeanSquaredError:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # 예측값
        self.t = None  # 실제값

    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = 0.5 * np.sum((y - t) ** 2) / y.shape[0]  # 평균 제곱 오차 계산
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dy = (self.y - self.t) * dout / batch_size  # 평균 제곱 오차의 기울기
        return dy
    
class EssayScorePredictor:
    def __init__(self, vocab_size, hidden_size, word_vecs):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        embed_W = word_vecs
        lstm_Wx = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (np.random.randn(H, 1) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(1).astype('f')

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = MeanSquaredError()

        # 모든 가중치와 기울기를 리스트에 모음
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    
    
def preprocess_paragraphs(paragraphs, word_to_id):
    processed_paragraphs = []
    for paragraph in paragraphs:
        words = paragraph.split()
        word_ids = [word_to_id.get(word, word_to_id['[UNK]']) for word in words]
        processed_paragraphs.append(word_ids)
    return processed_paragraphs

processed_paragraphs = preprocess_paragraphs(paragraphs, word_to_id)

# 모델 인스턴스 생성
model = EssayScorePredictor(word_vecs, hidden_size=100, output_size=1)

optimizer = Adam()


# 데이터셋 분할
def custom_train_test_split(data, labels, test_size=0.2, random_state=None):
    # 데이터 섞기
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))

    # 테스트 세트의 크기 계산
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # 데이터 분할
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]


def get_batch(data, scores, batch_size, shuffle=True):
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(data), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield [data[idx] for idx in batch_indices], scores[batch_indices]
        



def pad_sequences(sequences, maxlen=None, padding='post', padding_value=0):
    # 가장 긴 시퀀스의 길이 찾기
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # 결과를 저장할 numpy 배열 초기화
    padded_sequences = np.full((len(sequences), maxlen), padding_value)

    # 각 시퀀스에 대해 패딩 수행
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            # 시퀀스가 maxlen보다 길 경우, maxlen 길이까지 자름
            padded_sequences[i] = np.array(seq[:maxlen])
        else:
            # 시퀀스가 maxlen보다 짧은 경우, 패딩 추가
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            else:  # 'pre' padding
                padded_sequences[i, -len(seq):] = seq

    return padded_sequences

padded_paragraphs = pad_sequences(processed_paragraphs, max_sequence_length=235)
x_train, x_test, y_train, y_test = custom_train_test_split(padded_paragraphs, scores, test_size=0.2, random_state=42)


max_epoch = 10
batch_size = 32

# 데이터셋 크기
num_data = len(x_train)
max_iters = num_data // batch_size

for epoch in range(max_epoch):
    idx = np.random.permutation(num_data)
    x_train = x_train[idx]
    y_train = y_train[idx]

    for iters in range(max_iters):
        batch_x = x_train[iters*batch_size:(iters+1)*batch_size]
        batch_y = y_train[iters*batch_size:(iters+1)*batch_size].reshape(-1, 1)

        # Forward
        loss = model.forward(batch_x, batch_y)

        # Backward
        model.backward()
        optimizer.update(model.params, model.grads)

        # 간단한 로그 출력
        if iters % 10 == 0:
            print(f'Epoch {epoch+1}/{max_epoch}, Iteration {iters+1}/{max_iters}, Loss: {loss}')



# 검증 세트에서 성능 평가
def evaluate(model, x_test, y_test):
    scores = model.predict(x_test)
    loss = model.loss_layer.forward(scores, y_test.reshape(-1, 1))
    return loss

test_loss = evaluate(model, x_test, y_test)
print(f'Test Loss: {test_loss}')
