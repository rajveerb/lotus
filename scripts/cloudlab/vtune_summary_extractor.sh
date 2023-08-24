for i in 1 10 100;
do 
    vtune -report summary -report-output summary_${i}.csv -format csv -csv-delimiter comma -r /mydata/vtune_logs/pytorch_vtune_logs/vtune_vary_image_file_size_100Kfiles_v3/filesize${i}MB_b128_gpu4/;
done