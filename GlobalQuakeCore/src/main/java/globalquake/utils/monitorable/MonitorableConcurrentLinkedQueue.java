package globalquake.utils.monitorable;

import java.util.Collection;
import java.util.concurrent.ConcurrentLinkedQueue;

public class MonitorableConcurrentLinkedQueue<E> extends ConcurrentLinkedQueue<E> implements Monitorable {

    @SuppressWarnings("unused")
    public MonitorableConcurrentLinkedQueue(Collection<E> tmpList) {
        super(tmpList);
    }

    public MonitorableConcurrentLinkedQueue() {
        super();
    }

    @Override
    public boolean add(E e) {
        noteChange();
        return super.add(e);
    }


    @Override
    public boolean addAll(Collection<? extends E> c) {
        noteChange();
        return super.addAll(c);
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

}
