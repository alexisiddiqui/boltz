from pathlib import Path

from boltz.api import run_boltz


# Example usage
def main():
    # Define input data and output directory
    input_fasta = "/home/alexi/Documents/xFold_Sampling/boltz_sample/HOIP_dab3/HOIP_dab3.fasta"
    output_dir = "./results"

    # Run Boltz with custom parameters
    results = run_boltz(
        data=input_fasta,
        out_dir=output_dir,
        recycling_steps=4,  # Increase recycling steps for potentially better accuracy
        sampling_steps=100,  # Reduce sampling steps for faster prediction
        diffusion_samples=3,  # Generate multiple samples for each prediction
        step_scale=1.5,  # Adjust diffusion temperature
        output_format="pdb",  # Output in PDB format
        use_msa_server=True,  # Use MMSeqs2 server for MSA generation
        max_msa_seqs=2048,  # Limit MSA sequences for faster computation
        seed=42,  # Set seed for reproducibility
        return_results=True,  # Return results dictionary
    )

    # Access the results if return_results was True
    if results:
        print(f"Predictions saved to: {results['output_dir']}")
        print(
            f"Number of structures predicted: {len(results['predictions']) if results['predictions'] else 0}"
        )

        # Get the highest confidence structure's path (assumes prediction results contain this info)
        if results["predictions"]:
            # This is a placeholder - actual implementation would depend on the structure of results
            highest_conf_structure = (
                results["output_dir"] / "predictions" / f"{Path(input_fasta).stem}_model_0.pdb"
            )
            print(f"Highest confidence structure: {highest_conf_structure}")


if __name__ == "__main__":
    main()
