imageDir = 'test_images';
imageList = dir( fullfile( imageDir, '*.jpg'));

fid = fopen('test_images_gt.txt');
gt_bounding_boxes = textscan(fid, '%s %d %d %d %d');
fclose(fid);

nImages = length(imageList);
for i = 1:nImages
    close all
    imageName = imageList(i).name;
    fprintf('Visualizing faces in %s\n', imageName)
    im = im2double(imread(fullfile( imageDir, imageName)));
    if(size(im,3) > 1)
        im = rgb2gray(im);
    end
    
    faces = strcmp(imageName, gt_bounding_boxes{1,1});
    faces = find(faces);
    
    figure(1)
    imshow(im)
    hold on;
    for j = 1:length(faces)
        bbox = [gt_bounding_boxes{2}(faces(j)) ...
            gt_bounding_boxes{3}(faces(j)) ...
            gt_bounding_boxes{4}(faces(j)) ...
            gt_bounding_boxes{5}(faces(j))];
        
        plot_rectangle = [bbox(1), bbox(2); ...
            bbox(1), bbox(4); ...
            bbox(3), bbox(4); ...
            bbox(3), bbox(2); ...
            bbox(1), bbox(2)];
        plot( plot_rectangle(:,1), plot_rectangle(:,2) , 'g-')
    end
    pause;
end