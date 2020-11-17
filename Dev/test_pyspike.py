import matplotlib.pyplot as plt
import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("/home/william/repos/PySpike/test/PySpike_testdata.txt",
                                              edges=(0, 4000))
avrg_isi_profile = spk.isi_profile(spike_trains)
avrg_spike_profile = spk.spike_profile(spike_trains)
avrg_spike_sync_profile = spk.spike_sync_profile(spike_trains)

isi_profile = spk.isi_profile(spike_trains[0], spike_trains[1])
x, y = isi_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("ISI distance: %.8f" % isi_profile.avrg())
plt.show()

spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
x, y = spike_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("SPIKE distance: %.8f" % spike_profile.avrg())
plt.show()

spike_sync_profile = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
x, y = spike_sync_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("SPIKE SYNC distance: %.8f" % spike_sync_profile.avrg())
plt.show()
