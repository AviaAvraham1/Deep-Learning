from utils2 import load_model

" this file will load the saved models and evaluate them " ################# Evaluate model

# TODO: structure and implement the evaluation code here

    # print("Evaluating Trained Model...")
    # evaluate_classifier(encoder, classifier, test_loader, args)
    # print("Generating t-SNE Visualizations...")
    # plot_tsne(encoder, test_loader, args)
    
    # if args.self_supervised:
    #     print("Generating Reconstruction Images...")
    #     visualize_reconstructions(encoder, decoder, test_loader, args)

    # # Simple test to verify model integration with dataset
    # print("Running simple test on dataset with models...")
    # sample_images, _ = next(iter(train_loader))
    # sample_images = sample_images.view(sample_images.size(0), -1).to(args.device)
    
    # encoded = encoder(sample_images)
    # print(f"Encoder output shape: {encoded.shape} (Expected: {sample_images.size(0)} x {args.latent_dim})")
    
    # if args.self_supervised:
    #     reconstructed = decoder(encoded)
    #     print(f"Decoder output shape: {reconstructed.shape} (Expected: {sample_images.size(0)} x {sample_images.shape[1]})")
    
    # class_preds = classifier(encoded)
    # print(f"Classifier output shape: {class_preds.shape} (Expected: {sample_images.size(0)} x {NUM_CLASSES})")
    
    # print("✅ Model integration test passed!")