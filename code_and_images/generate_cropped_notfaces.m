% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

while n_have < n_want
    rand_img = floor((height(imageList)-1) * rand() + 1);
    temp_img = imread(strcat(imageList(rand_img).folder, "/",imageList(rand_img).name));
    temp_img = rgb2gray(temp_img);
    [h, w] = size(temp_img);
    rand_height = floor((h-37) * rand() + 1);
    rand_width = floor((w-37) * rand() + 1);
    gen_img = temp_img(rand_height:rand_height+36, rand_width:rand_width+36);
    imwrite(gen_img, strcat(new_imageDir, "/", int2str(n_have), ".jpg"));
    n_have = n_have + 1;
end
%%
new_imageDir2 = 'cropped_validation_images_face';
new_imageDir3 = 'cropped_validation_images_notface';
mkdir(new_imageDir2);
mkdir(new_imageDir3);
valid_size = round(n_want * (0.2));
imageDir = 'cropped_training_images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
imageDir2 = 'cropped_training_images_faces';
imageList2 = dir(sprintf('%s/*.jpg',imageDir2));
for i=1:1:valid_size
   movefile(strcat(imageList(i).folder, "/",imageList(i).name), 'cropped_validation_images_notface');
   movefile(strcat(imageList2(i).folder, "/",imageList2(i).name), 'cropped_validation_images_face');
end
