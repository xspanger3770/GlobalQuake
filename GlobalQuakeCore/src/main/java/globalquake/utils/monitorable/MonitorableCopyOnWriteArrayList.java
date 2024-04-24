package globalquake.utils.monitorable;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class MonitorableCopyOnWriteArrayList<E> extends CopyOnWriteArrayList<E> implements Monitorable {

    @SuppressWarnings("unused")
    public MonitorableCopyOnWriteArrayList(List<E> tmpList) {
        super(tmpList);
    }

    public MonitorableCopyOnWriteArrayList() {
        super();
    }

    @Override
    public boolean add(E e) {
        noteChange();
        return super.add(e);
    }

    @Override
    public void add(int index, E element) {
        noteChange();
        super.add(index, element);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        noteChange();
        return super.addAll(c);
    }

    @Override
    public boolean addAll(int index, Collection<? extends E> c) {
        noteChange();
        return super.addAll(index, c);
    }

    @Override
    public void clear() {
        noteChange();
        super.clear();
    }

    @Override
    public boolean remove(Object o) {
        noteChange();
        return super.remove(o);
    }

    @Override
    public E set(int index, E element) {
        noteChange();
        return super.set(index, element);
    }
}
