from OgreInterface.generate import InterfaceGenerator, SurfaceGenerator
from OgreInterface.miller import MillerSearch
from pymatgen.io.vasp.inputs import Poscar


# A Miller index scan can be performed as follows
ms = MillerSearch(
    substrate="./poscars/POSCAR_InAs_conv",
    film="./poscars/POSCAR_Al_conv",
    max_film_index=2,
    max_substrate_index=2,
    max_strain=0.01,
    max_area_mismatch=0.01,
    max_area=200
)
ms.run_scan()
ms.plot_misfits(fontsize=14, labelrotation=-20, figure_scale=1)

# Generate a list of InAs(111) Surface classes with different terminations
subs = SurfaceGenerator.from_file(
    "./poscars/POSCAR_InAs_conv",
    miller_index=[1, 1, 1],
    layers=5,
    vacuum=10,
)

# Generate a list of Al(111) Surface classes with different terminations
films = SurfaceGenerator.from_file(
    "./poscars/POSCAR_Al_conv",
    miller_index=[1, 1, 1],
    layers=5,
    vacuum=10,
)

# Select which Surface you want from the list of Surface's
sub = subs._slabs[0]
film = films._slabs[0]

# Optional way to remove layers from the substrate or film (Usefull for complete control of termination)
#  sub.remove_layers(num_layers=3, top=False)
#  film.remove_layers(num_layers=1, top=True)

# Set up the interface generator
interface_generator = InterfaceGenerator(
    substrate=sub,
    film=film,
    max_strain=0.01,
    max_area_mismatch=0.01,
    max_area=200,
    interfacial_distance=4.0,
    vacuum=20,
    center=True,
)

# Generate the interfaces
interfaces = interface_generator.generate_interfaces()

# Loop through interfaces to do whatever you want with them
for i, interface in enumerate(interfaces):
    # If you print the interface you will get all of the important information printed out
    print(interface)
    print("")

    # Plot how the unit cells line up to form a commensurate interface
    interface.plot_interface(output=f"interface_view_{i:02d}.png")

    # Write the interface structure
    interface.write_file(output=f"POSCAR_{i:02d}", orthogonal=True)
    # Poscar(interface.interface).write_file(f"POSCAR_{i:02d}")
