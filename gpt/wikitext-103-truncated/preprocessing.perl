#!/usr/bin/perl
use strict;
use warnings;
use utf8;
use open ':std', ':encoding(UTF-8)';

sub process_dataset {
    my ($input_file_path, $output_file_path) = @_;
    open my $file, '<', $input_file_path or die "Could not open input file: $!";
    
    my $long_text = join '', <$file>;
    close $file;

    $long_text = lc($long_text);
    $long_text =~ s/[^abcdefghijklmnopqrstuvwxyz .,]//g;
    $long_text =~ s/\s+/ /g;

    open my $out_file, '>', $output_file_path or die "Could not open output file: $!";
    print $out_file $long_text;
    close $out_file;
}

# Usage
my $input_file_path = 'wiki.test.raw';
my $output_file_path = 'wiki_shannonfied.test.txt';
process_dataset($input_file_path, $output_file_path);

my $input_file_path = 'wiki.valid.raw';
my $output_file_path = 'wiki_shannonfied.valid.txt';
process_dataset($input_file_path, $output_file_path);
