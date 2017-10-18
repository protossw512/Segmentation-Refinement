files = dir('./DAVIS-2017-trainval-480p/DAVIS/SegPredictions/*.mat');

for file = files'
   file_path = strcat(file.folder, '/', file.name);
   index = strfind(file.name,'_');
   class_name = file.name(1:index-1);
   data = load(file_path);
   for object = 1:size(data.choice,1)
       channel = data.choice(object);
       if isfield(data, 'pixel_level')
           for frame = 2:length(data.pixel_level)
               mask = data.pixel_level{frame}(:,:,channel);
               foldername = strcat('./DAVIS-2017-trainval-480p/DAVIS/SegPredictions/', ... 
                   class_name, '/', int2str(object));
               if ~exist(foldername, 'dir')
                   mkdir(foldername)
               end
               filename = strcat(foldername, '/', sprintf('%05d', frame), '.png');
               imwrite(mask, filename);
               disp(filename);
           end
       end
   end
end