{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab69c109",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request, jsonify\n",
    "import torch\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "import os\n",
    "\n",
    "class SimpleLSTMModel(torch.nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, output_size):\n",
    "        super(SimpleLSTMModel, self).__init__()\n",
    "        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)\n",
    "        self.fc = torch.nn.Linear(hidden_size, output_size)\n",
    "\n",
    "    def forward(self, x):\n",
    "        lstm_out, _ = self.lstm(x)\n",
    "        output = self.fc(lstm_out[:, -1, :])\n",
    "        return output\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load the pre-trained model\n",
    "model = SimpleLSTMModel(input_size=1, hidden_size=32, output_size=1)\n",
    "model.load_state_dict(torch.load('simple_lstm_model.pth'))\n",
    "model.eval()\n",
    "\n",
    "# Load the CSV data\n",
    "csv_path = os.path.join(os.getcwd(), \"NFLX.csv\") \n",
    "data = pd.read_csv(csv_path)\n",
    "target_column = \"Close\"\n",
    "target = data[target_column].values.reshape(-1, 1)\n",
    "\n",
    "# Normalize the target values\n",
    "scaler = MinMaxScaler(feature_range=(-1, 1))\n",
    "target_normalized = scaler.fit_transform(target)\n",
    "\n",
    "# Prepare input sequences\n",
    "sequence_length = 10\n",
    "data_sequences = torch.tensor([], dtype=torch.float32)\n",
    "target_sequences = torch.tensor([], dtype=torch.float32)\n",
    "\n",
    "for i in range(len(target_normalized) - sequence_length):\n",
    "    data_seq = torch.tensor(target_normalized[i:i+sequence_length], dtype=torch.float32)\n",
    "    target_seq = torch.tensor(target_normalized[i+sequence_length:i+sequence_length+1], dtype=torch.float32)\n",
    "\n",
    "    data_sequences = torch.cat((data_sequences, data_seq.unsqueeze(0)), dim=0)\n",
    "    target_sequences = torch.cat((target_sequences, target_seq.unsqueeze(0)), dim=0)\n",
    "\n",
    "# Split the data into training and testing sets\n",
    "train_size = int(len(data_sequences) * 0.80)\n",
    "test_size = len(data_sequences) - train_size\n",
    "train_data, test_data = torch.utils.data.random_split(data_sequences, [train_size, test_size])\n",
    "train_target, test_target = torch.utils.data.random_split(target_sequences, [train_size, test_size])\n",
    "\n",
    "# Convert to tensors\n",
    "train_data = torch.stack([item[0] for item in train_data])\n",
    "test_data = torch.stack([item[0] for item in test_data])\n",
    "train_target = torch.stack([item for item in train_target])\n",
    "test_target = torch.stack([item for item in test_target])\n",
    "\n",
    "# Create DataLoader\n",
    "batch_size = 64\n",
    "train_dataloader = DataLoader(TensorDataset(train_data, train_target), batch_size=batch_size, shuffle=True)\n",
    "test_dataloader = DataLoader(TensorDataset(test_data, test_target), batch_size=batch_size, shuffle=False)\n",
    "\n",
    "# Routes\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        data = request.get_json()\n",
    "        input_data = torch.tensor(data['input_data'], dtype=torch.float32).unsqueeze(0).unsqueeze(2)\n",
    "\n",
    "        with torch.no_grad():\n",
    "            prediction = model(input_data)\n",
    "\n",
    "        # Inverse transform the prediction\n",
    "        prediction = scaler.inverse_transform(prediction.numpy().reshape(-1, 1)).flatten()\n",
    "\n",
    "        return jsonify({'prediction': prediction.tolist()})\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)})\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True, use_reloader=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd992015",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
