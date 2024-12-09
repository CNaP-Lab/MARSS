function run_image_list = spm5_image_list(num_vols, file_list)
    num_runs = length(num_vols);

    if(length(file_list) == num_runs)
        % EITHER cell array with full image list OR single 4-D image names

        for i = 1:num_runs
            if size(file_list{i}, 1) == num_vols(i)
                % we have the full list already % tor edit april 2010
                for j = 1:num_vols(i)
                    run_image_list{i}{j} = deblank(file_list{i}(j, :));
                end
            else
                % it's a single image name; expand it and create names
                % (could use expand_4d_images as well)
                printf_str = ['%' int2str(size(int2str(max(num_vols)), 2)) 'd'];
                for j = 1:num_vols(i)
                    run_image_list{i}{j} = [deblank(file_list{i}) ',' num2str(j)];
                end
            end
        end
        
    else
        % another format; not cell array of length(nruns)
        st = [0 cumsum(num_vols(1:end-1))];
        for i = 1:num_runs
            for j = 1:num_vols(i)
                run_image_list{i}{j} = [file_list{st(i) + j} ',1'];
            end
        end
    end
end