# %% [markdown]
# # Dementia Prediction
# - **Dementia** is one of the common causes of memory loss and death gloabally. Many people are not even aware of the fact that they might have dementia or not. Hence, we are driven by this motivation to build machine learning applications that have no unbiasedness and possess high probability of detecting and predicting dementia at any stage.

# %% [markdown]
# #### Introduction 
# > This project leverages deep learning models(CNN models) for the prediction of Dementia or no Dementia amongst well represented patients samples. 
# > This project will make use of pytorch and different CNN models. 
# - The best model will be deployed for real usage in medical sectors for dementia prediction 

# %% [markdown]
# #### Importing Libraries 

# %%
#import methods from the torchvision class
from torchvision.datasets import ImageFolder 
from torchvision import transforms , datasets 
from torch.utils.data import DataLoader, random_split 
import matplotlib.pyplot as plt 
import os 
import torch.nn as nn 

# %%
print(os.path.exists('Alzheimer_dataset/train'))

# %%
print(os.listdir('Alzheimer_dataset/train'))

# %% [markdown]
# ### Prior Examination and transformation 
# #### Transformation 

# %%

#use the transforms object to apply the needed transformation by accessing the Compose class 
#using the transforms object
transform_traindata = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(128, 128)),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((128, 128))
])


#use only the three transform methods
transform_testdata= transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     transforms.Resize((128, 128))
])

# %% [markdown]
# #### Data Collection 

# %%
train_data = ImageFolder(
    #use the "r" to make python avoid escape issues
  r"C:\Users\512GB\OneDrive\Documents\Research Projects\Dementia Research\Deep-Learning-and-Classical-ML-models-for-Dementia-Prediction\Predicting Dementia\Alzheimer_dataset\train",
  transform = transform_traindata
)

train_size= int(0.8 * len(train_data))
val_size= len(train_data) - train_size

train_dataset, val_dataset= random_split(train_data, [train_size, val_size])

val_dataset.transform= transform_testdata



test = ImageFolder(
    #use the "r" to make python avoid escape issues
  r"C:\Users\512GB\OneDrive\Documents\Research Projects\Dementia Research\Deep-Learning-and-Classical-ML-models-for-Dementia-Prediction\Predicting Dementia\Alzheimer_dataset\test",
  transform = transform_testdata
)   


# %%
#instantiate an object of the data loader class 
dataloader_train = DataLoader(
  train_dataset, shuffle=True, batch_size=4, num_workers=2
)

dataloader_val = DataLoader(
    val_dataset, shuffle= False, batch_size=2, num_workers=2
     
)

dataloader_test= DataLoader(
    test, shuffle=False, batch_size=2, num_workers=2
) 

# %%
print(len(dataloader_train))
print(len(dataloader_test))
print(len(dataloader_val))

# %%
#get an image and its label
image, label = next(iter(dataloader_train))
#Reshape the image tensor
print(image.shape ) 


# %%
#image = image.squeeze().permute(1, 2, 0)

# %%
import matplotlib.pyplot as plt

# Define the figure (2 rows with 5 images each) and axis objects
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
# Flatten the axes array
axs = axs.ravel()

# Initialize index for subplots
subplot_index = 0

# Load first 10 examples
for images, labels in dataloader_train:
    if subplot_index >= 10:  # Only plot up to 10 images
        break

    # Plot individual images from the batch
    for j in range(images.size(0)):  # Loop through the batch
        if subplot_index >= 10:  # Stop if we have plotted 10 images
            break
        
        # Extract the j-th image and label from the batch
        image = images[j]  # shape [3, 128, 128]
        label = labels[j]  # Assuming labels is of the same size as images

        # Plot image and set title to the corresponding label
        axs[subplot_index].imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
        axs[subplot_index].title.set_text(f'Label: {"Demented" if label.item() == 0 else "Non-Demented"}')
        axs[subplot_index].axis('off')  # Hide axes for visual appeal

        subplot_index += 1  # Move to the next subplot

plt.tight_layout()
plt.show();


# %% [markdown]
# #### Feature Extraction and fully connected layers

# %%
#create a class of the model on the nn.Module parent class

class Net(nn.Module):
    """
    Define the arguments and parameters hers 
    Define the arguments and parameters hers
    Define the arguments and parameters hers
    Define the arguments and parameters hers
    Define the arguments and parameters hers
    """
    #initialize the class constructor and input self and needed parameters
    def __init__(self, num_classes):
        #connec the class with the properties and methods using the super function
        super().__init__()
        #define the sequential method to extract the needed features
        self.feature_extractor= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),


            nn.Flatten() 

        )

        self.imageclassifier = nn.Sequential( 
            nn.Linear(128 , 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        


    def forward(self, x):
            x= self.feature_extractor(x)
            x= self.imageclassifier(x)
            return x 


# %%
model_inspection = Net(num_classes=2)
print(model_inspection)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to draw a rectangle with text
def draw_rectangle(ax, xy, width, height, text, fontsize=10):
    rect = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.3", 
                                  edgecolor="black", facecolor="lightgray")
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", 
            fontsize=fontsize, color="black")

# Initialize the plot
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')  # Turn off the axes

# Draw the rectangles for each layer with more spacing and larger font
layer_width = 3
layer_height = 1.5

draw_rectangle(ax, (1, 9), layer_width, layer_height, "Input\n(3 channels)", fontsize=12)
draw_rectangle(ax, (5, 9), layer_width, layer_height, "Conv2D\n(3, 32)", fontsize=12)
draw_rectangle(ax, (9, 9), layer_width, layer_height, "LeakyReLU", fontsize=12)
draw_rectangle(ax, (13, 9), layer_width, layer_height, "BatchNorm2D\n(32)", fontsize=12)
draw_rectangle(ax, (17, 9), layer_width, layer_height, "MaxPool2D", fontsize=12)

draw_rectangle(ax, (5, 7), layer_width, layer_height, "Conv2D\n(32, 64)", fontsize=12)
draw_rectangle(ax, (9, 7), layer_width, layer_height, "LeakyReLU", fontsize=12)
draw_rectangle(ax, (13, 7), layer_width, layer_height, "BatchNorm2D\n(64)", fontsize=12)
draw_rectangle(ax, (17, 7), layer_width, layer_height, "MaxPool2D", fontsize=12)

draw_rectangle(ax, (5, 5), layer_width, layer_height, "Conv2D\n(64, 128)", fontsize=12)
draw_rectangle(ax, (9, 5), layer_width, layer_height, "LeakyReLU", fontsize=12)
draw_rectangle(ax, (13, 5), layer_width, layer_height, "BatchNorm2D\n(128)", fontsize=12)
draw_rectangle(ax, (17, 5), layer_width, layer_height, "MaxPool2D", fontsize=12)

draw_rectangle(ax, (9, 3), layer_width, layer_height, "Flatten", fontsize=12)
draw_rectangle(ax, (13, 3), layer_width, layer_height, "Linear\n(128*8*8, 256)", fontsize=12)
draw_rectangle(ax, (17, 3), layer_width, layer_height, "LeakyReLU", fontsize=12)
draw_rectangle(ax, (13, 1), layer_width, layer_height, "Dropout\n(0.5)", fontsize=12)
draw_rectangle(ax, (17, 1), layer_width, layer_height, "Linear\n(256, num_classes)", fontsize=12)

# Draw arrows between layers
arrow_props = dict(arrowstyle="->", linewidth=2, color="black")

for start, end in [(4, 9.75), (8, 9.75), (12, 9.75), (16, 9.75),
                   (4, 7.75), (8, 7.75), (12, 7.75), (16, 7.75),
                   (4, 5.75), (8, 5.75), (12, 5.75), (16, 5.75),
                   (10.5, 4.5), (14.5, 4.5), (14.5, 2.5), (14.5, 2)]:
    ax.annotate("", xy=(end, start), xytext=(start-2.5, start),
                arrowprops=arrow_props)

# Display the plot
plt.show()


# %% [markdown]
# #### Training Loop 

# %%
import torch.optim as optim

#initialize the Net class
net = Net(num_classes=2)

#initailize the cross entropy class
criterion= nn.CrossEntropyLoss()

#optimizer
optimizer1 = optim.Adam(net.parameters(), lr=0.0005)


for epoch in range(50): 
    running_loss=0
    for images, labels in dataloader_train:
        optimizer1.zero_grad()
        outputs= net(images)
        loss= criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        running_loss += loss.item()

    epoch_loss= running_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# %% [markdown]
# optimizer2 = optim.SGD(net.parameters(), lr=0.7, momentum=0.9)
# for epoch in range(50):
#     running_loss=0
#     for images, labels in dataloader_train:
#         optimizer2.zero_grad()
#         outputs= net(images)
#         loss= criterion(outputs, labels)
#         loss.backward()
#         optimizer1.step()
#         running_loss += loss.item()
# 
#         epoch_loss= running_loss / len(dataloader_train)
#         print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
# 


