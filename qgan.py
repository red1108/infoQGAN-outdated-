class QGAN():

    def __init__(self):
        
    def generator_init(generator_input):
        for i in range(n_qubits):
            qml.RY((generator_input[i]-0.5) * np.pi/4, wires=i)
        

    def generator_layer(params):
        for i in range(n_qubits):
            qml.RY(params[i][0], wires=i)
        for i in range(n_qubits):
            qml.CZ(wires=[i, (i+1)%n_qubits])



    @qml.qnode(dev, interface="torch")
    def generator_circuit(params, generator_input):
        generator_init(generator_input)
        for param in params[:n_layers]:
            generator_layer(param)
        for i in range(n_qubits):
            qml.RY(params[n_layers][0], wires=i)

        return qml.probs(wires=range(output_qubits))


    def generator_forward(params, generator_input):
        generator_output = [generator_circuit(params, single_in) for single_in in generator_input] # (BATCH_SIZE, 2**output_qubits) 차원
        generator_output = torch.stack(generator_output) # (BATCH_SIZE, 2**output_qubits) 차원
        return generator_output


    def generator_train_step(params, generator_input, use_mine = False, _qmine = False):
        code_input = generator_input[:, -code_qubits:]

        generator_output = generator_forward(params, generator_input)
        generator_output = generator_output.to(torch.float32) # (BATCH_SIZE, output_qubits)
        
        disc_output = discriminator(generator_output) # 밑에 코드에서 정의됨
        gan_loss = -torch.log(disc_output).mean()
        

        if use_mine:
            pred_xy = mine(code_input, generator_output)
            code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
            pred_x_y = mine(code_input_shuffle, generator_output)
            mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            gan_loss -= coeff * mi


        return generator_output, gan_loss
