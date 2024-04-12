# Run :  python RandSampler.py -s "data/test/mix1.gz" "data/test/mix2.gz" "data/test/mix3.gz" -d "data/test-mixed" -p 0.0 0.3 0.7 -t 6

import argparse
import json
import multiprocessing
import random
from contextlib import ExitStack
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import msgspec
import smart_open
import uniseg.wordbreak
from dolma.core.data_types import InputSpec
from dolma.core.parallel import BaseParallelProcessor
from dolma.core.paths import join_path
from dolma.core.runtime import _make_paths_from_prefix

class RandomSampler(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue[Union[Tuple[int, ...], None]],
        /,
        files: int = 0,
        documents: int = 0,
        words: int = 0,
        extracted: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue, files=files, documents=documents, extracted=extracted, uniseg_words=words
        )

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        decoder = msgspec.json.Decoder(InputSpec)
       
        probability = kwargs.get("probability", None)
        if probability is None:
            raise ValueError("Probability must be specified")
        probability = float(probability)
        print(f"INSIDE RANDOM SAMPLER PROB = {probability}")
        print(f"SAMPLER SOURCE = {source_path}")
        complement = kwargs.get("complement", False)
        unique_samples = set()

        extracted_count = documents_count = words_count = 0
        update_interval = 1

        with ExitStack() as stack:
            source = stack.enter_context(smart_open.open(source_path, "rt"))
            destination = stack.enter_context(smart_open.open(destination_path, "wt"))

            total_documents = sum(1 for _ in source)  # Count total documents in the dataset
            num_samples = int(probability)  # Calculate number of samples to select
   
            # Reset file pointer to the beginning of the file
            source.seek(0)

            for line in source:
                data = decoder.decode(line)

                # Check if the line should be included based on the sampling probability
                if len(unique_samples) < num_samples and random.random() < probability:
                    if data.text not in unique_samples:
                        extracted_count += 1
                        words_count += sum(1 for w in uniseg.wordbreak.words(data.text.strip()) if w.strip())
                        destination.write(line)
                        unique_samples.add(data.text)

                documents_count += 1
                if documents_count % update_interval == 0:
                    # Update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(
                        queue, documents=documents_count, extracted=extracted_count, words=words_count
                    )
                    
                    extracted_count = documents_count = words_count = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # Double the update interval if the queue is full
                        update_interval *= 2
          
        cls.increment_progressbar(
            queue, files=1, documents=documents_count, extracted=extracted_count, words=words_count
        )

def main(
    sources: List[str],
    destination: str,
    probabilities: List[float],
    total_samples: int,
    complement: bool = False,
    num_workers: int = 1,
    debug: bool = False,
    dryrun: bool = False,
) -> None:
    if len(sources) != len(probabilities):
        raise ValueError("Number of source datasets must match number of probabilities")

    if total_samples <= 0:
        raise ValueError("Total samples must be a positive integer")

    print(f"Sampling with probabilities: {probabilities}")
    print(f"Total samples: {total_samples}")

    if dryrun:
        return

    with TemporaryDirectory() as tempdir:
        dest_prefixes = _make_paths_from_prefix(sources, join_path(None, destination))
        meta_prefixes = _make_paths_from_prefix(sources, join_path(None, tempdir))

        processor_instances = []
        total_probability = sum(probabilities)
        calculated_samples = [int(total_samples * prob / total_probability) for prob in probabilities]
        total_calculated_samples = sum(calculated_samples)
        diff_samples = total_samples - total_calculated_samples
        
        # Adjust the number of samples so that the total matches total_samples
        for i in range(len(calculated_samples)):
            if diff_samples > 0:
                calculated_samples[i] += 1
                diff_samples -= 1
            elif diff_samples < 0:
                calculated_samples[i] -= 1
                diff_samples += 1

        for src, num_samples, dest_prefix, meta_prefix in zip(sources, calculated_samples, dest_prefixes, meta_prefixes):
            processor = RandomSampler(
                source_prefix=src,
                destination_prefix=dest_prefix,
                metadata_prefix=meta_prefix,
                num_processes=num_workers,
                debug=debug,
            )
            processor_instances.append((processor, src, dest_prefix, num_samples, complement))

        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(run_processor, processor_instances)

def run_processor(processor_instance, source_path, destination_path, num_samples, complement):
    processor_instance.process_single(source_path, destination_path, queue=Queue(), probability=num_samples, complement=complement)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sources", type=str, required=True, help="List of input files", nargs="+")
    ap.add_argument("-d", "--destination", type=str, required=True, help="Destination directory")
    ap.add_argument("-p", "--probabilities", type=float, required=True, help="List of sampling probabilities", nargs="+")
    ap.add_argument("-n", "--num-workers", type=int, default=1, help="Number of worker processes")
    ap.add_argument("-t", "--total-samples", type=int, required=True, help="Total number of samples")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode")
    ap.add_argument("--dryrun", action="store_true", help="Perform a dry run without actual processing")
    ap.add_argument("--complement", action="store_true", help="Get the complement of the sample")
    opts = ap.parse_args()

    print(json.dumps(vars(opts), indent=2, sort_keys=True))
    return opts

if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
