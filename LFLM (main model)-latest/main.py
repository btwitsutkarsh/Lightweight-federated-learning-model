import torch
from torchvision import datasets, transforms
from models.net import Net
from src.client import Client
from src.server import Server
from src.parser import parse_args
from src.utils import setup_ssl_context, partition_data
from compression import compress_trained_model
import os
from src.federated_visualizer import FederatedLearningVisualizer
from src.neural_visualizer import create_neural_diagram, visualize_mnist_cnn

def main():
    # Get parsed arguments
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Setup SSL context for downloading dataset
    setup_ssl_context()

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("Loading MNIST dataset...")
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, download=True, transform=transform)

    # Create and visualize initial model
    initial_model = Net()
    print("\nGenerating network visualization...")
    visualize_mnist_cnn(initial_model, 'mnist_architecture')

    # Print initial model information
    print("\nInitial Model Information:\n")
    print("Model Architecture:")
    print(initial_model)

    # Partition data among clients
    print("\nPartitioning data among clients...")
    client_datasets = partition_data(trainset, args.num_clients)

    # Create clients
    clients = []
    for i in range(args.num_clients):
        model = Net()
        client = Client(i, model, client_datasets[i], args.batch_size, args.epochs, device, args.learning_rate, args.momentum)
        clients.append(client)

    # Create server
    server_model = Net()
    server = Server(server_model, testset, args.batch_size, device)

    # Training loop
    print("\nStarting Federated Learning training...")
    for round in range(args.num_rounds):
        print(f'\nRound {round + 1}/{args.num_rounds}')

        # Train clients
        client_models = []
        for client in clients:
            client_state_dict = client.train()
            client_models.append(client_state_dict)

        # Aggregate models
        print("\nAggregating models...")
        server.aggregate(client_models)

        # Test global model
        print("\nTesting global model:")
        test_loss, test_accuracy = server.test()
        
        print(f"\nRound {round + 1} Summary:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Create directory for saved models if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Save the trained federated model
    save_path = 'saved_models/federated_model.pth'
    torch.save({
        'model_state_dict': server_model.state_dict(),
        'final_test_accuracy': test_accuracy,
        'num_rounds': args.num_rounds,
        'num_clients': args.num_clients
    }, save_path)
    
    print(f"\nTrained federated model saved to: {save_path}")

    # Compress the trained model
    print("\nCompressing the trained model...")
    compressed_model = compress_trained_model(
        model_path=save_path,
        device=device,
        test_dataset=testset,
        batch_size=args.batch_size,
        compression_rate=args.pruning_amount  # Changed to match the parameter name
    )

    # Generate visualization of compressed model
    print("\nGenerating compressed model visualization...")
    visualize_mnist_cnn(compressed_model, 'compressed_mnist_architecture')

if __name__ == '__main__':
    main()