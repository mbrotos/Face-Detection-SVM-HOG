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