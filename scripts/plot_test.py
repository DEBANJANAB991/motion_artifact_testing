import matplotlib.pyplot as plt

# inside evaluate(), after loading model:
for i, (art, clean) in enumerate(test_loader):
    if i >= 3: break
    art, clean = art.to(device), clean.to(device)
    pred = model(art)
    art, clean, pred = art[0,0].cpu(), clean[0,0].cpu(), pred[0,0].cpu()

    fig, axs = plt.subplots(1,3, figsize=(9,3))
    axs[0].imshow(art,  cmap='gray'); axs[0].set_title('Artifact')
    axs[1].imshow(pred, cmap='gray'); axs[1].set_title('Reconstructed')
    axs[2].imshow(clean,cmap='gray'); axs[2].set_title('Ground Truth')
    for ax in axs: ax.axis('off')
    plt.show()
